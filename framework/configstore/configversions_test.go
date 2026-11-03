package configstore

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testProviderConfigWithKey(keyName, keyValue string) ProviderConfig {
	secret := schemas.NewSecretVar(keyValue)
	return ProviderConfig{
		Keys: []schemas.Key{
			// The transport layer assigns key IDs before calling the store;
			// tests use the key name as a deterministic unique ID.
			{ID: "key-" + keyName, Name: keyName, Value: *secret, Models: []string{"gpt-4"}},
		},
	}
}

// TestGetProviderConfigVersions_EmptyDB pins that probing an empty database
// succeeds with empty results — the worker's first tick must not error.
func TestGetProviderConfigVersions_EmptyDB(t *testing.T) {
	s := setupRDBTestStore(t)
	ctx := context.Background()

	providers, err := s.GetProviderConfigVersions(ctx)
	require.NoError(t, err)
	assert.Empty(t, providers)

	vks, err := s.GetVirtualKeyVersions(ctx)
	require.NoError(t, err)
	assert.Empty(t, vks)
}

// TestGetProviderConfigVersions_ProviderAndKeys pins the probe shape: one
// entry per provider carrying its own updated_at plus a per-key (key_id,
// updated_at) set.
func TestGetProviderConfigVersions_ProviderAndKeys(t *testing.T) {
	s := setupRDBTestStore(t)
	ctx := context.Background()

	require.NoError(t, s.AddProvider(ctx, schemas.OpenAI, testProviderConfigWithKey("openai-key", "sk-test")))

	versions, err := s.GetProviderConfigVersions(ctx)
	require.NoError(t, err)
	require.Len(t, versions, 1)
	v := versions[0]
	assert.Equal(t, schemas.OpenAI, v.Provider)
	assert.False(t, v.UpdatedAt.IsZero())
	require.Len(t, v.Keys, 1)
	assert.NotEmpty(t, v.Keys[0].KeyID)
	assert.Equal(t, string(schemas.OpenAI), v.Keys[0].Provider)
	assert.False(t, v.Keys[0].UpdatedAt.IsZero())
}

// TestGetProviderConfigVersions_KeyUpdateDoesNotBumpProvider pins the design
// premise of carrying per-key timestamps: key writes touch only config_keys,
// so the provider row's updated_at does not move. If this changes (some
// future write path starts bumping the parent row), the probe stays correct —
// but configsync's key-set diff must keep comparing keys, not rely on it.
func TestGetProviderConfigVersions_KeyUpdateDoesNotBumpProvider(t *testing.T) {
	s := setupRDBTestStore(t)
	ctx := context.Background()

	require.NoError(t, s.AddProvider(ctx, schemas.OpenAI, testProviderConfigWithKey("openai-key", "sk-one")))
	before, err := s.GetProviderConfigVersions(ctx)
	require.NoError(t, err)
	require.Len(t, before, 1)

	// Ensure the provider-row timestamp and the key write are distinguishable.
	time.Sleep(10 * time.Millisecond)
	updated := testProviderConfigWithKey("openai-key", "sk-two")
	updated.Keys[0].ID = before[0].Keys[0].KeyID
	require.NoError(t, s.UpdateProviderKey(ctx, schemas.OpenAI, before[0].Keys[0].KeyID, updated.Keys[0]))

	after, err := s.GetProviderConfigVersions(ctx)
	require.NoError(t, err)
	require.Len(t, after, 1)
	// Key token moved…
	assert.False(t, after[0].Keys[0].UpdatedAt.Equal(before[0].Keys[0].UpdatedAt),
		"key update must bump the key's own updated_at")
	// …provider token must NOT move on a key-only write.
	assert.True(t, after[0].UpdatedAt.Equal(before[0].UpdatedAt),
		"key-only writes must not bump the provider row's updated_at")
}

// TestGetProviderConfigVersions_DeletionVanishes pins deletion detection: a
// provider absent from the probe was deleted from the database.
func TestGetProviderConfigVersions_DeletionVanishes(t *testing.T) {
	s := setupRDBTestStore(t)
	ctx := context.Background()

	require.NoError(t, s.AddProvider(ctx, schemas.OpenAI, testProviderConfigWithKey("openai-key", "sk-one")))
	require.NoError(t, s.AddProvider(ctx, schemas.Groq, testProviderConfigWithKey("groq-key", "gq-one")))
	require.NoError(t, s.DeleteProvider(ctx, schemas.OpenAI))

	versions, err := s.GetProviderConfigVersions(ctx)
	require.NoError(t, err)
	require.Len(t, versions, 1)
	assert.Equal(t, schemas.Groq, versions[0].Provider)
}

// TestGetVirtualKeyVersions_CoversTouchAndDelete pins the VK probe plus
// TouchVirtualKey — the escape hatch for child-row writes that must become
// visible to VK-row change detection.
func TestGetVirtualKeyVersions_CoversTouchAndDelete(t *testing.T) {
	s := setupRDBTestStore(t)
	ctx := context.Background()

	vkID := "vk-probe-test"
	vk := &tables.TableVirtualKey{
		ID:       vkID,
		Name:     "Probe Test VK",
		Value:    *schemas.NewSecretVar("vk-probe-test-value"),
		IsActive: schemas.Ptr(true),
	}
	require.NoError(t, s.CreateVirtualKey(ctx, vk))

	before, err := s.GetVirtualKeyVersions(ctx)
	require.NoError(t, err)
	require.Len(t, before, 1)
	assert.Equal(t, vkID, before[0].ID)

	time.Sleep(10 * time.Millisecond)
	require.NoError(t, s.TouchVirtualKey(ctx, vkID))

	after, err := s.GetVirtualKeyVersions(ctx)
	require.NoError(t, err)
	require.Len(t, after, 1)
	assert.False(t, after[0].UpdatedAt.Equal(before[0].UpdatedAt),
		"TouchVirtualKey must bump the VK row's updated_at")

	require.NoError(t, s.DeleteVirtualKey(ctx, vkID))
	empty, err := s.GetVirtualKeyVersions(ctx)
	require.NoError(t, err)
	assert.Empty(t, empty)
}

// TestTouchVirtualKey_NotFound pins the error contract for a missing row.
func TestTouchVirtualKey_NotFound(t *testing.T) {
	s := setupRDBTestStore(t)
	err := s.TouchVirtualKey(context.Background(), "no-such-vk")
	assert.ErrorIs(t, err, ErrNotFound)
}
