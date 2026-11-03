package server

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
)

// configVersionStore is the store capability the reconcile worker needs:
// cheap (id, updated_at) probes over the watched config tables. Asserted
// against configstore.ConfigStore rather than added to it — the main interface
// is very large and every implementor (including the test mock) would have to
// grow. Both sqlite and postgres stores are *RDBConfigStore and satisfy this;
// same capability-assertion pattern as logstore.LogRetentionManager.
type configVersionStore interface {
	GetProviderConfigVersions(ctx context.Context) ([]configstore.ProviderConfigVersion, error)
	GetVirtualKeyVersions(ctx context.Context) ([]configstore.VirtualKeyVersion, error)
}

// providerWatermark is the previous tick's change token for one provider and
// its keys. Doubles as the row-set member used for deletion detection: a
// provider present in the watermark map but absent from the probe was deleted
// from the database.
type providerWatermark struct {
	updatedAt time.Time
	keys      map[string]time.Time
}

// configSyncWorker periodically reconciles in-memory providers/keys and
// virtual keys from the config database so mutations made on one OSS node
// (via its HTTP API) become visible on peer nodes within one interval.
// Enterprise deployments converge via cluster gossip instead and never start
// this worker (see Bootstrap).
//
// Change detection is timestamp-based: each tick probes (id, updated_at) for
// the watched tables and diffs against the previous tick's snapshot. Entities
// whose tokens moved are re-applied through the existing mutation paths with
// BifrostContextKeySkipDBUpdate=true — the worker never writes to the
// database. client.UpdateProvider rebuilds a provider's connection pools, so
// unchanged entities are never touched.
//
// Accepted behaviors (discussed in the design; see docs):
//   - A local API edit is re-applied once on the next tick (its own write
//     bumped updated_at). Cost: one redundant provider rebuild per edit.
//   - Any node restart bumps every provider's updated_at (startup re-upserts
//     all providers), so peers re-apply all providers once per restart.
//   - A local edit landing between the worker's DB read and its memory apply
//     can be transiently reverted for at most one interval.
type configSyncWorker struct {
	server   *BifrostHTTPServer
	interval time.Duration
	versions configVersionStore

	// Snapshots from the last tick; owned by the run goroutine only.
	providers         map[schemas.ModelProvider]providerWatermark
	virtualKeys       map[string]time.Time
	seededProviders   bool
	seededVirtualKeys bool

	// governanceLoaded caches whether the governance plugin is loaded — the
	// VK apply/remove paths require it. Providers sync regardless.
	governanceLoaded bool

	stopCh   chan struct{}
	stopOnce sync.Once
	cancel   context.CancelFunc
}

// configSyncReloadTimeout bounds one provider's governance+catalog reload,
// mirroring the handler path's discovery budget (handlers/providers.go).
const configSyncReloadTimeout = 15 * time.Second

// newConfigSyncWorker returns nil (with a one-line Info reason) when the
// worker must not run: config_sync not enabled, no config store, store lacks
// the version probes, or source_of_truth=config.json (that mode's restart
// file→DB sync can delete DB-only providers and fights periodic reconcile).
func newConfigSyncWorker(s *BifrostHTTPServer) *configSyncWorker {
	if s.Config == nil {
		return nil
	}
	if !s.Config.ConfigSync.IsEnabled() {
		logger.Info("config sync disabled (config_sync.enabled not set in config.json)")
		return nil
	}
	if s.Config.ConfigStore == nil {
		logger.Info("config sync disabled (no config store)")
		return nil
	}
	if s.Config.IsConfigJSONSourceOfTruth() {
		logger.Info("config sync disabled (source_of_truth=config.json owns reconciliation)")
		return nil
	}
	versions, ok := s.Config.ConfigStore.(configVersionStore)
	if !ok {
		logger.Info("config sync disabled (config store does not support version probes)")
		return nil
	}
	return &configSyncWorker{
		server:           s,
		interval:         s.Config.ConfigSync.IntervalOrDefault(),
		versions:         versions,
		providers:        make(map[schemas.ModelProvider]providerWatermark),
		virtualKeys:      make(map[string]time.Time),
		governanceLoaded: s.Config.IsPluginLoaded(s.getGovernancePluginName()),
		stopCh:           make(chan struct{}),
	}
}

func (w *configSyncWorker) start(ctx context.Context) {
	runCtx, cancel := context.WithCancel(ctx)
	w.cancel = cancel
	go w.run(runCtx)
}

func (w *configSyncWorker) stop() {
	w.stopOnce.Do(func() {
		// Cancel in-flight DB reads so shutdown is not held up by them, then
		// signal run() to exit its ticker loop. Like the oauth2 sweeper this
		// does not join the goroutine — an apply already past its DB read may
		// still complete against the shutting-down server.
		if w.cancel != nil {
			w.cancel()
		}
		close(w.stopCh)
	})
}

func (w *configSyncWorker) run(ctx context.Context) {
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	w.reconcile(ctx)
	for {
		select {
		case <-ticker.C:
			w.reconcile(ctx)
		case <-w.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// reconcile runs one pass. A panicking entity must not kill the goroutine —
// a silently-dead background worker would quietly desync the node — so the
// whole pass is wrapped in a recover.
func (w *configSyncWorker) reconcile(ctx context.Context) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("config sync pass panicked: %v", r)
		}
	}()
	w.reconcileProviders(ctx)
	w.reconcileVirtualKeys(ctx)
}

func (w *configSyncWorker) reconcileProviders(ctx context.Context) {
	versions, err := w.versions.GetProviderConfigVersions(ctx)
	if err != nil {
		logger.Warn("config sync: provider version probe failed: %v", err)
		return
	}
	if !w.seededProviders {
		w.providers = foldProviderVersions(versions)
		w.seededProviders = true
		// Boot-loaded memory matches the database by construction (LoadConfig
		// reads it), but a peer write landing between LoadConfig and this
		// first tick leaves membership gaps. Close those without rebuilding
		// providers present in both — content equality at boot is guaranteed,
		// and re-applying every provider on every worker start would churn
		// every connection pool.
		w.reconcileProviderMembership(ctx, w.providers)
		return
	}
	changed, removed := diffProviderVersions(w.providers, versions)
	for _, provider := range removed {
		skipCtx := context.WithValue(ctx, schemas.BifrostContextKeySkipDBUpdate, true)
		if err := w.server.RemoveProvider(skipCtx, provider); err != nil {
			// Keep the watermark so the removal is retried next tick.
			logger.Warn("config sync: remove provider %s failed: %v", provider, err)
		} else {
			delete(w.providers, provider)
			logger.Info("config sync: removed provider %s (deleted in database)", provider)
		}
	}
	for _, provider := range changed {
		w.applyProvider(ctx, provider, watermarkOf(versions, provider))
	}
}

// applyProvider re-reads the provider from the database and applies it to
// memory and the core client. The watermark is only advanced on full success,
// so a failed apply is retried on the next tick.
func (w *configSyncWorker) applyProvider(ctx context.Context, provider schemas.ModelProvider, mark providerWatermark) {
	full, err := w.server.Config.ConfigStore.GetProviderConfig(ctx, provider)
	if err != nil {
		logger.Warn("config sync: read provider %s failed: %v", provider, err)
		return
	}
	skipCtx := context.WithValue(ctx, schemas.BifrostContextKeySkipDBUpdate, true)

	_, memErr := w.server.Config.GetProviderConfigRaw(provider)
	switch {
	case memErr == nil:
		err = w.server.Config.UpdateProviderConfig(skipCtx, provider, *full)
	case errors.Is(memErr, lib.ErrNotFound):
		// Upsert idiom (mirrors handlers/providers.go): tolerate a concurrent
		// local mutation flipping existence between the check and the apply.
		err = w.server.Config.AddProvider(skipCtx, provider, *full)
		if errors.Is(err, lib.ErrAlreadyExists) {
			err = w.server.Config.UpdateProviderConfig(skipCtx, provider, *full)
		}
		if err == nil {
			// AddProvider does not touch the core client; UpdateProviderConfig
			// did it for the update branch, do it for the add branch here.
			if clientErr := w.server.Client.UpdateProvider(provider); clientErr != nil {
				logger.Warn("config sync: client update for new provider %s failed: %v", provider, clientErr)
			}
		}
	default:
		err = memErr
	}
	if err != nil {
		logger.Warn("config sync: apply provider %s failed: %v", provider, err)
		return
	}

	// Governance + model catalog sync. skipUsageWriteback: this node's
	// in-memory usage counters are authoritative for this node; writing them
	// back on reconcile would make the DB usage columns oscillate across
	// nodes. Rare by construction — only runs when updated_at actually moved.
	reloadCtx, cancel := context.WithTimeout(ctx, configSyncReloadTimeout)
	if _, err := w.server.reloadProvider(reloadCtx, provider, true); err != nil {
		// Non-fatal: memory is already applied; catalog may lag until the
		// next change or a peer's live-model refresher.
		logger.Warn("config sync: governance/catalog reload for provider %s failed: %v", provider, err)
	}
	cancel()

	w.providers[provider] = mark
	logger.Info("config sync: applied provider %s from database", provider)
}

// reconcileProviderMembership applies the membership-only diff between the
// freshly seeded watermarks and in-memory providers: a DB row with no
// in-memory counterpart is applied (a peer created it during our boot), and
// an in-memory provider with no DB row is removed (a peer deleted it during
// our boot).
func (w *configSyncWorker) reconcileProviderMembership(ctx context.Context, inDB map[schemas.ModelProvider]providerWatermark) {
	for provider, mark := range inDB {
		if _, err := w.server.Config.GetProviderConfigRaw(provider); errors.Is(err, lib.ErrNotFound) {
			w.applyProvider(ctx, provider, mark)
		}
	}
	for _, provider := range w.memoryProviders() {
		if _, ok := inDB[provider]; !ok {
			skipCtx := context.WithValue(ctx, schemas.BifrostContextKeySkipDBUpdate, true)
			if err := w.server.RemoveProvider(skipCtx, provider); err != nil {
				logger.Warn("config sync: remove provider %s failed: %v", provider, err)
			} else {
				delete(w.providers, provider)
				logger.Info("config sync: removed provider %s (deleted in database)", provider)
			}
		}
	}
}

// memoryProviders lists in-memory providers under Config.Mu, sorted for
// deterministic apply order.
func (w *configSyncWorker) memoryProviders() []schemas.ModelProvider {
	w.server.Config.Mu.RLock()
	providers := make([]schemas.ModelProvider, 0, len(w.server.Config.Providers))
	for provider := range w.server.Config.Providers {
		providers = append(providers, provider)
	}
	w.server.Config.Mu.RUnlock()
	sort.Slice(providers, func(i, j int) bool { return providers[i] < providers[j] })
	return providers
}

func (w *configSyncWorker) reconcileVirtualKeys(ctx context.Context) {
	versions, err := w.versions.GetVirtualKeyVersions(ctx)
	if err != nil {
		logger.Warn("config sync: virtual key version probe failed: %v", err)
		return
	}
	if !w.seededVirtualKeys {
		w.virtualKeys = foldVirtualKeyVersions(versions)
		w.seededVirtualKeys = true
		logger.Info("config sync: seeded %d providers, %d virtual keys", len(w.providers), len(w.virtualKeys))
		// Same boot-race closing as providers, restricted to the apply
		// direction: a VK row the governance store does not know (created by
		// a peer during our boot) is loaded now. The reverse — a stale VK
		// lingering in memory after a peer's boot-time delete — has no cheap
		// enumeration (the governance store exposes only point lookups) and
		// converges on the row's next edit.
		if w.governanceLoaded {
			if governancePlugin, err := w.server.getGovernancePlugin(); err == nil {
				govStore := governancePlugin.GetGovernanceStore()
				for _, v := range versions {
					if _, ok := govStore.GetVirtualKeyByID(ctx, v.ID); !ok {
						if _, err := w.server.ReloadVirtualKey(ctx, v.ID); err != nil {
							logger.Warn("config sync: seed reload virtual key %s failed: %v", v.ID, err)
						}
					}
				}
			}
		}
		return
	}
	if !w.governanceLoaded {
		return
	}
	changed, removed := diffVirtualKeyVersions(w.virtualKeys, versions)
	for _, id := range removed {
		if err := w.server.RemoveVirtualKey(ctx, id); err != nil {
			logger.Warn("config sync: remove virtual key %s failed: %v", id, err)
		} else {
			delete(w.virtualKeys, id)
			logger.Info("config sync: removed virtual key %s (deleted in database)", id)
		}
	}
	for _, id := range changed {
		// ReloadVirtualKey re-reads the VK and its scoped model configs from
		// the DB and applies them through the usage-preserving in-memory
		// upserts; it performs no DB writes.
		if _, err := w.server.ReloadVirtualKey(ctx, id); err != nil {
			logger.Warn("config sync: reload virtual key %s failed: %v", id, err)
		} else {
			w.virtualKeys[id] = watermarkOfVK(versions, id)
		}
	}
}

// foldProviderVersions builds the seed snapshot from a probe result.
func foldProviderVersions(versions []configstore.ProviderConfigVersion) map[schemas.ModelProvider]providerWatermark {
	folded := make(map[schemas.ModelProvider]providerWatermark, len(versions))
	for _, v := range versions {
		folded[v.Provider] = watermarkFromVersion(v)
	}
	return folded
}

func watermarkFromVersion(v configstore.ProviderConfigVersion) providerWatermark {
	keys := make(map[string]time.Time, len(v.Keys))
	for _, k := range v.Keys {
		keys[k.KeyID] = k.UpdatedAt
	}
	return providerWatermark{updatedAt: v.UpdatedAt, keys: keys}
}

func watermarkOf(versions []configstore.ProviderConfigVersion, provider schemas.ModelProvider) providerWatermark {
	for _, v := range versions {
		if v.Provider == provider {
			return watermarkFromVersion(v)
		}
	}
	return providerWatermark{}
}

func foldVirtualKeyVersions(versions []configstore.VirtualKeyVersion) map[string]time.Time {
	folded := make(map[string]time.Time, len(versions))
	for _, v := range versions {
		folded[v.ID] = v.UpdatedAt
	}
	return folded
}

func watermarkOfVK(versions []configstore.VirtualKeyVersion, id string) time.Time {
	for _, v := range versions {
		if v.ID == id {
			return v.UpdatedAt
		}
	}
	return time.Time{}
}

// diffProviderVersions returns the providers whose change token moved (new,
// provider-level change, or key-set/key-level change) and those that vanished
// from the probe since the previous snapshot. Both slices are sorted for
// deterministic apply order. Equality (not ordering) is compared, so clock
// skew between writers cannot hide a change.
func diffProviderVersions(prev map[schemas.ModelProvider]providerWatermark, cur []configstore.ProviderConfigVersion) (changed, removed []schemas.ModelProvider) {
	for _, v := range cur {
		old, ok := prev[v.Provider]
		if !ok || !old.updatedAt.Equal(v.UpdatedAt) || !keysEqual(old.keys, v.Keys) {
			changed = append(changed, v.Provider)
		}
	}
	for provider := range prev {
		found := false
		for _, v := range cur {
			if v.Provider == provider {
				found = true
				break
			}
		}
		if !found {
			removed = append(removed, provider)
		}
	}
	sort.Slice(changed, func(i, j int) bool { return changed[i] < changed[j] })
	sort.Slice(removed, func(i, j int) bool { return removed[i] < removed[j] })
	return changed, removed
}

func keysEqual(prev map[string]time.Time, cur []configstore.KeyConfigVersion) bool {
	if len(prev) != len(cur) {
		return false
	}
	for _, k := range cur {
		old, ok := prev[k.KeyID]
		if !ok || !old.Equal(k.UpdatedAt) {
			return false
		}
	}
	return true
}

// diffVirtualKeyVersions is the VK analogue of diffProviderVersions.
func diffVirtualKeyVersions(prev map[string]time.Time, cur []configstore.VirtualKeyVersion) (changed, removed []string) {
	for _, v := range cur {
		old, ok := prev[v.ID]
		if !ok || !old.Equal(v.UpdatedAt) {
			changed = append(changed, v.ID)
		}
	}
	for id := range prev {
		found := false
		for _, v := range cur {
			if v.ID == id {
				found = true
				break
			}
		}
		if !found {
			removed = append(removed, id)
		}
	}
	sort.Strings(changed)
	sort.Strings(removed)
	return changed, removed
}
