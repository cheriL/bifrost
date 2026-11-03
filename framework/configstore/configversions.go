package configstore

import (
	"context"
	"sort"
	"time"

	"gorm.io/gorm"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore/tables"
)

// KeyConfigVersion is the change token for one provider key row, keyed by the
// schemas.Key UUID column (config_keys.key_id), not the table's numeric PK.
type KeyConfigVersion struct {
	Provider  string
	KeyID     string
	UpdatedAt time.Time
}

// ProviderConfigVersion is the change token for one provider row plus its keys.
// Key CRUD (CreateProviderKey/UpdateProviderKey/DeleteProviderKey) writes only
// config_keys and never bumps the provider row, so keys carry their own
// timestamps and are folded in here.
type ProviderConfigVersion struct {
	Provider  schemas.ModelProvider
	UpdatedAt time.Time
	Keys      []KeyConfigVersion
}

// VirtualKeyVersion is the change token for one virtual key row. It
// deliberately excludes budget/rate-limit child rows: the governance usage
// tracker dumps counters into governance_budgets/governance_rate_limits every
// few seconds, so their updated_at is noise, while every VK-mutating endpoint
// persists the VK row itself.
type VirtualKeyVersion struct {
	ID        string
	UpdatedAt time.Time
}

// GetProviderConfigVersions returns (provider, updated_at) plus per-key
// (key_id, updated_at) for the whole config_providers/config_keys tables.
//
// CHANGE-DETECTION CONTRACT: the config sync worker in
// transports/bifrost-http/server/configsync.go detects cross-node changes by
// comparing these timestamps. Any new write path that mutates
// config_providers, config_keys, or governance_virtual_keys must go through a
// gorm write that auto-updates UpdatedAt (Save/Update/Create/OnConflict
// UpdateAll). Raw UpdateColumn/UpdateColumns calls skip the auto-timestamp and
// would make the change invisible to every other node in a clustered
// deployment. If a child-row write must stay invisible-cheap (like the webhook
// operational counters), that is fine — but a *config* change routed around
// the parent row needs to touch the parent row's updated_at in the same
// transaction (see the VK-row touch in mutateVirtualKeyBudgetOverride and the
// model-config handlers for precedent).
func (s *RDBConfigStore) GetProviderConfigVersions(ctx context.Context) ([]ProviderConfigVersion, error) {
	type providerRow struct {
		Name      string    `gorm:"column:name"`
		UpdatedAt time.Time `gorm:"column:updated_at"`
	}
	var providerRows []providerRow
	// Table().Select(...).Scan(...) into a local struct: no model hooks run,
	// no secrets are decrypted, only the two indexed columns are read.
	if err := s.DB().WithContext(ctx).
		Table("config_providers").
		Select("name, updated_at").
		Scan(&providerRows).Error; err != nil {
		return nil, err
	}

	type keyRow struct {
		Provider  string    `gorm:"column:provider"`
		KeyID     string    `gorm:"column:key_id"`
		UpdatedAt time.Time `gorm:"column:updated_at"`
	}
	var keyRows []keyRow
	if err := s.DB().WithContext(ctx).
		Table("config_keys").
		Select("config_providers.name AS provider, config_keys.key_id, config_keys.updated_at").
		Joins("JOIN config_providers ON config_providers.id = config_keys.provider_id").
		Scan(&keyRows).Error; err != nil {
		return nil, err
	}

	keysByProvider := make(map[schemas.ModelProvider][]KeyConfigVersion, len(providerRows))
	for _, row := range keyRows {
		keysByProvider[schemas.ModelProvider(row.Provider)] = append(
			keysByProvider[schemas.ModelProvider(row.Provider)],
			KeyConfigVersion{Provider: row.Provider, KeyID: row.KeyID, UpdatedAt: row.UpdatedAt},
		)
	}

	versions := make([]ProviderConfigVersion, 0, len(providerRows))
	for _, row := range providerRows {
		provider := schemas.ModelProvider(row.Name)
		keys := keysByProvider[provider]
		// Deterministic order so callers comparing successive snapshots never
		// see ordering-only churn.
		sort.Slice(keys, func(i, j int) bool { return keys[i].KeyID < keys[j].KeyID })
		versions = append(versions, ProviderConfigVersion{
			Provider:  provider,
			UpdatedAt: row.UpdatedAt,
			Keys:      keys,
		})
	}
	sort.Slice(versions, func(i, j int) bool { return versions[i].Provider < versions[j].Provider })
	return versions, nil
}

// TouchVirtualKey bumps the virtual key row's updated_at so VK-row change
// detection observes writes that would otherwise land only on child rows
// (budget overrides, VK-scoped model configs). Callers should invoke it in the
// same transaction as the child-row write — see the change-detection contract
// on GetProviderConfigVersions.
func (s *RDBConfigStore) TouchVirtualKey(ctx context.Context, vkID string, tx ...*gorm.DB) error {
	db := s.DB()
	if len(tx) > 0 {
		db = tx[0]
	}
	res := db.WithContext(ctx).
		Model(&tables.TableVirtualKey{}).
		Where("id = ?", vkID).
		Update("updated_at", time.Now())
	if res.Error != nil {
		return res.Error
	}
	if res.RowsAffected == 0 {
		return ErrNotFound
	}
	return nil
}

// GetVirtualKeyVersions returns (id, updated_at) for all virtual keys. See the
// change-detection contract on GetProviderConfigVersions.
func (s *RDBConfigStore) GetVirtualKeyVersions(ctx context.Context) ([]VirtualKeyVersion, error) {
	type vkRow struct {
		ID        string    `gorm:"column:id"`
		UpdatedAt time.Time `gorm:"column:updated_at"`
	}
	var vkRows []vkRow
	if err := s.DB().WithContext(ctx).
		Table("governance_virtual_keys").
		Select("id, updated_at").
		Scan(&vkRows).Error; err != nil {
		return nil, err
	}
	versions := make([]VirtualKeyVersion, 0, len(vkRows))
	for _, row := range vkRows {
		versions = append(versions, VirtualKeyVersion{ID: row.ID, UpdatedAt: row.UpdatedAt})
	}
	sort.Slice(versions, func(i, j int) bool { return versions[i].ID < versions[j].ID })
	return versions, nil
}
