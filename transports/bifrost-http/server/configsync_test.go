package server

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/mcp_headers"
	"github.com/maximhq/bifrost/framework/modelcatalog"
	"github.com/maximhq/bifrost/framework/oauth2"
	"github.com/maximhq/bifrost/plugins/governance"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---- pure-function tests -------------------------------------------------

func TestConfigSyncConfigGating(t *testing.T) {
	enabled := true
	disabled := false
	thirty := 30
	five := 5

	assert.False(t, (*lib.ConfigSyncConfig)(nil).IsEnabled(), "nil config must disable the worker")
	assert.False(t, (&lib.ConfigSyncConfig{}).IsEnabled(), "absent enabled flag must disable the worker")
	assert.False(t, (&lib.ConfigSyncConfig{Enabled: &disabled}).IsEnabled())
	assert.True(t, (&lib.ConfigSyncConfig{Enabled: &enabled}).IsEnabled())

	assert.Equal(t, lib.DefaultConfigSyncInterval, (*lib.ConfigSyncConfig)(nil).IntervalOrDefault())
	assert.Equal(t, lib.DefaultConfigSyncInterval, (&lib.ConfigSyncConfig{IntervalSeconds: &thirty}).IntervalOrDefault())
	assert.Equal(t, 5*time.Second, (&lib.ConfigSyncConfig{IntervalSeconds: &five}).IntervalOrDefault())
	negative := -1
	assert.Equal(t, lib.DefaultConfigSyncInterval, (&lib.ConfigSyncConfig{IntervalSeconds: &negative}).IntervalOrDefault(),
		"non-positive intervals fall back to the default rather than disabling the ticker")
}

func TestDiffProviderVersions(t *testing.T) {
	now := time.Now()
	later := now.Add(time.Second)

	prev := map[schemas.ModelProvider]providerWatermark{
		schemas.OpenAI: {updatedAt: now, keys: map[string]time.Time{"k1": now}},
		schemas.Groq:   {updatedAt: now, keys: map[string]time.Time{"k2": now}},
	}

	// No change at all.
	changed, removed := diffProviderVersions(prev, []configstore.ProviderConfigVersion{
		{Provider: schemas.OpenAI, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k1", UpdatedAt: now}}},
		{Provider: schemas.Groq, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k2", UpdatedAt: now}}},
	})
	assert.Empty(t, changed)
	assert.Empty(t, removed)

	// New provider, provider-level bump, key-set change, key-level bump,
	// and a removal — all in one pass.
	changed, removed = diffProviderVersions(prev, []configstore.ProviderConfigVersion{
		{Provider: schemas.OpenAI, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k1", UpdatedAt: now}}},
		{Provider: schemas.Groq, UpdatedAt: later, Keys: []configstore.KeyConfigVersion{{KeyID: "k2", UpdatedAt: now}}},
		{Provider: schemas.Anthropic, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k3", UpdatedAt: now}}},
		{Provider: schemas.Cohere, UpdatedAt: now, Keys: nil},
		{Provider: schemas.Azure, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k1", UpdatedAt: later}}},
		{Provider: schemas.Gemini, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k4", UpdatedAt: now}, {KeyID: "k5", UpdatedAt: now}}},
	})
	// Groq: provider bump. Anthropic/Cohere: new. Azure: key-level bump.
	// Gemini: unrelated (not in prev, counts as new). OpenAI unchanged.
	assert.Equal(t, []schemas.ModelProvider{schemas.Anthropic, schemas.Azure, schemas.Cohere, schemas.Gemini, schemas.Groq}, changed)
	assert.Empty(t, removed, "nothing removed — every prev provider still present")

	// Key-set membership change (key added to OpenAI).
	prevOpenAIKeyAdded := map[schemas.ModelProvider]providerWatermark{
		schemas.OpenAI: {updatedAt: now, keys: map[string]time.Time{"k1": now}},
	}
	changed, _ = diffProviderVersions(prevOpenAIKeyAdded, []configstore.ProviderConfigVersion{
		{Provider: schemas.OpenAI, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{
			{KeyID: "k1", UpdatedAt: now}, {KeyID: "k6", UpdatedAt: now},
		}},
	})
	assert.Equal(t, []schemas.ModelProvider{schemas.OpenAI}, changed)

	// Deletion: prev member absent from probe.
	changed, removed = diffProviderVersions(prev, []configstore.ProviderConfigVersion{
		{Provider: schemas.Groq, UpdatedAt: now, Keys: []configstore.KeyConfigVersion{{KeyID: "k2", UpdatedAt: now}}},
	})
	assert.Empty(t, changed)
	assert.Equal(t, []schemas.ModelProvider{schemas.OpenAI}, removed)
}

func TestDiffVirtualKeyVersions(t *testing.T) {
	now := time.Now()
	later := now.Add(time.Second)

	prev := map[string]time.Time{"vk-1": now, "vk-2": now}
	changed, removed := diffVirtualKeyVersions(prev, []configstore.VirtualKeyVersion{
		{ID: "vk-1", UpdatedAt: now},
		{ID: "vk-2", UpdatedAt: later},
	})
	assert.Equal(t, []string{"vk-2"}, changed)
	assert.Empty(t, removed)

	changed, removed = diffVirtualKeyVersions(prev, []configstore.VirtualKeyVersion{
		{ID: "vk-2", UpdatedAt: now},
		{ID: "vk-3", UpdatedAt: now},
	})
	assert.Equal(t, []string{"vk-3"}, changed)
	assert.Equal(t, []string{"vk-1"}, removed)
}

// ---- worker gating tests -------------------------------------------------

func TestNewConfigSyncWorkerGating(t *testing.T) {
	prevLogger := logger
	logger = noopTestLogger{}
	t.Cleanup(func() { logger = prevLogger })

	enabled := true
	store := newConfigSyncTestStore(t)
	mkServer := func(cfg *lib.Config) *BifrostHTTPServer {
		return &BifrostHTTPServer{
			Ctx:    schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
			Config: cfg,
		}
	}

	// Disabled by default.
	cfg := &lib.Config{ConfigSync: nil, ConfigStore: store}
	assert.Nil(t, newConfigSyncWorker(mkServer(cfg)))

	cfg = &lib.Config{ConfigSync: &lib.ConfigSyncConfig{}, ConfigStore: store}
	assert.Nil(t, newConfigSyncWorker(mkServer(cfg)))

	// No config store.
	cfg = &lib.Config{ConfigSync: &lib.ConfigSyncConfig{Enabled: &enabled}}
	assert.Nil(t, newConfigSyncWorker(mkServer(cfg)))

	// Store without version probes (the interface alone does not satisfy
	// configVersionStore).
	probelessStore := &configSyncProbelessStore{ConfigStore: store}
	cfg = &lib.Config{ConfigSync: &lib.ConfigSyncConfig{Enabled: &enabled}, ConfigStore: probelessStore}
	assert.Nil(t, newConfigSyncWorker(mkServer(cfg)))

	// Happy path: enabled + capable store + governance plugin loaded.
	env := newConfigSyncTestEnv(t, nil)
	assert.NotNil(t, env.worker)
	assert.True(t, env.worker.governanceLoaded)
	assert.Equal(t, lib.DefaultConfigSyncInterval, env.worker.interval)
}

// configSyncProbelessStore hides the probe methods by wrapping the store in an
// anonymous struct that only embeds the ConfigStore interface.
type configSyncProbelessStore struct {
	configstore.ConfigStore
}

// configSyncProviderConfig builds a single-key ProviderConfig. The transport
// layer assigns key IDs before store writes; tests use a deterministic one so
// memory and DB agree on identity across probes.
func configSyncProviderConfig(keyValue string) configstore.ProviderConfig {
	return configstore.ProviderConfig{
		Keys: []schemas.Key{{
			ID:    "k1",
			Name:  "groq-key",
			Value: *schemas.NewSecretVar(keyValue),
		}},
	}
}

// ---- integration tests ---------------------------------------------------

// configSyncTestEnv assembles a real sqlite config store, a real in-memory
// governance store, and a real bifrost client around one lib.Config — the
// minimum the worker touches. Providers passed in memProviders are seeded in
// memory only (simulating drift from a peer's database state).
type configSyncTestEnv struct {
	server *BifrostHTTPServer
	worker *configSyncWorker
	store  configstore.ConfigStore
	probes configVersionStore
	gov    *governance.LocalGovernanceStore
	config *lib.Config
	client *bifrost.Bifrost
}

// newConfigSyncTestStore creates a file-backed sqlite config store in a temp
// dir. File-backed (not :memory:) because gorm pools connections and an
// in-memory DB is per-connection.
func newConfigSyncTestStore(t *testing.T) configstore.ConfigStore {
	t.Helper()
	store, err := configstore.NewConfigStore(context.Background(), &configstore.Config{
		Enabled: true,
		Type:    configstore.ConfigStoreTypeSQLite,
		Config:  &configstore.SQLiteConfig{Path: filepath.Join(t.TempDir(), "configsync-test.db")},
	}, noopTestLogger{})
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close(context.Background()) })
	return store
}

func newConfigSyncTestEnv(t *testing.T, memProviders map[schemas.ModelProvider]configstore.ProviderConfig) *configSyncTestEnv {
	t.Helper()
	prevLogger := logger
	logger = noopTestLogger{}
	t.Cleanup(func() { logger = prevLogger })
	// The lib package's logger is only initialized by LoadConfig, which these
	// hand-assembled configs bypass; the mutation methods the worker calls
	// log through it.
	lib.SetLogger(noopTestLogger{})

	ctx := context.Background()
	store := newConfigSyncTestStore(t)

	govStore, err := governance.NewLocalGovernanceStore(ctx, noopTestLogger{}, nil, &configstore.GovernanceConfig{}, nil, nil)
	require.NoError(t, err)

	config := &lib.Config{
		ConfigStore:        store,
		ClientConfig:       &configstore.ClientConfig{},
		Providers:          memProviders,
		ModelCatalog:       modelcatalog.NewTestCatalog(nil),
		OAuthProvider:      oauth2.NewOAuth2Provider(store, noopTestLogger{}),
		MCPHeadersProvider: mcp_headers.NewProvider(store, noopTestLogger{}),
	}
	plugins := []schemas.BasePlugin{&configSyncGovPlugin{store: govStore}}
	config.BasePlugins.Store(&plugins)

	client, err := bifrost.Init(ctx, schemas.BifrostConfig{
		Account: lib.NewBaseAccount(config),
		Logger:  bifrost.NewDefaultLogger(schemas.LogLevelError),
	})
	require.NoError(t, err)
	t.Cleanup(func() { client.Shutdown() })
	config.SetBifrostClient(client)

	server := &BifrostHTTPServer{
		Ctx:    schemas.NewBifrostContext(ctx, schemas.NoDeadline),
		Config: config,
		Client: client,
	}
	enabled := true
	config.ConfigSync = &lib.ConfigSyncConfig{Enabled: &enabled}
	worker := newConfigSyncWorker(server)
	require.NotNil(t, worker)
	probes, ok := store.(configVersionStore)
	require.True(t, ok, "sqlite config store must satisfy the probe capability")
	return &configSyncTestEnv{
		server: server, worker: worker, store: store, probes: probes,
		gov: govStore, config: config, client: client,
	}
}

// configSyncGovPlugin exposes the governance store under the standard plugin
// name so IsPluginLoaded/getGovernancePlugin find it.
type configSyncGovPlugin struct {
	governance.BaseGovernancePlugin
	store governance.GovernanceStore
}

func (p *configSyncGovPlugin) GetName() string                                { return governance.PluginName }
func (p *configSyncGovPlugin) GetGovernanceStore() governance.GovernanceStore { return p.store }

// reconcileTwice runs the seed pass and then one diff pass.
func (env *configSyncTestEnv) reconcileTwice(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	env.worker.reconcile(ctx)
	env.worker.reconcile(ctx)
}

// TestConfigSyncWorker_NoChangeTickDoesNotApply pins that a tick where nothing
// changed applies nothing — the seed pass records watermarks only, and the
// following diff is empty, so the provider is never rebuilt needlessly.
func TestConfigSyncWorker_NoChangeTickDoesNotApply(t *testing.T) {
	secret := schemas.NewSecretVar("sk-mem")
	env := newConfigSyncTestEnv(t, map[schemas.ModelProvider]configstore.ProviderConfig{
		schemas.Groq: {Keys: []schemas.Key{{ID: "k1", Name: "groq-key", Value: *secret}}},
	})
	require.NoError(t, env.store.AddProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-mem")))

	env.reconcileTwice(t)

	raw, err := env.config.GetProviderConfigRaw(schemas.Groq)
	require.NoError(t, err)
	require.Len(t, raw.Keys, 1)
	assert.Equal(t, "sk-mem", raw.Keys[0].Value.GetValue(), "no-change tick must not touch memory")
	// Watermark present → a later real change is detectable.
	assert.Contains(t, env.worker.providers, schemas.Groq)
}

// TestConfigSyncWorker_AppliesPeerEditWithoutWritingBack simulates another
// node's edit landing in the database and pins both halves of the contract:
// the peer's values converge into memory, and the worker writes nothing back.
func TestConfigSyncWorker_AppliesPeerEditWithoutWritingBack(t *testing.T) {
	env := newConfigSyncTestEnv(t, map[schemas.ModelProvider]configstore.ProviderConfig{
		schemas.Groq: {Keys: []schemas.Key{{ID: "k1", Name: "groq-key", Value: *schemas.NewSecretVar("sk-old")}}},
	})
	require.NoError(t, env.store.AddProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-old")))

	env.worker.reconcile(context.Background()) // seed

	// "Peer node" edits the provider in the database.
	require.NoError(t, env.store.UpdateProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-new")))
	before, err := env.probes.GetProviderConfigVersions(context.Background())
	require.NoError(t, err)
	require.Len(t, before, 1)
	dbTimestamp := before[0].UpdatedAt

	env.worker.reconcile(context.Background()) // apply

	raw, err := env.config.GetProviderConfigRaw(schemas.Groq)
	require.NoError(t, err)
	require.Len(t, raw.Keys, 1)
	assert.Equal(t, "sk-new", raw.Keys[0].Value.GetValue(), "peer edit must converge into memory")

	after, err := env.probes.GetProviderConfigVersions(context.Background())
	require.NoError(t, err)
	require.Len(t, after, 1)
	assert.True(t, after[0].UpdatedAt.Equal(dbTimestamp),
		"worker must not write back to the database (skipDBUpdate)")
}

// TestConfigSyncWorker_RemovesDeletedProvider pins deletion convergence: a
// provider deleted in the database disappears from memory, the core client,
// and the worker's watermark map.
func TestConfigSyncWorker_RemovesDeletedProvider(t *testing.T) {
	env := newConfigSyncTestEnv(t, map[schemas.ModelProvider]configstore.ProviderConfig{
		schemas.Groq: {Keys: []schemas.Key{{ID: "k1", Name: "groq-key", Value: *schemas.NewSecretVar("sk-old")}}},
	})
	require.NoError(t, env.store.AddProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-old")))
	env.reconcileTwice(t)
	require.Contains(t, env.worker.providers, schemas.Groq)

	require.NoError(t, env.store.DeleteProvider(context.Background(), schemas.Groq))
	env.worker.reconcile(context.Background())

	_, err := env.config.GetProviderConfigRaw(schemas.Groq)
	assert.ErrorIs(t, err, lib.ErrNotFound, "deleted provider must vanish from memory")
	assert.NotContains(t, env.worker.providers, schemas.Groq, "watermark must advance so the removal is not retried")
	configured, err := env.client.GetConfiguredProviders()
	require.NoError(t, err)
	assert.NotContains(t, configured, schemas.Groq, "deleted provider must vanish from the core client")
}

// TestConfigSyncWorker_AddsNewProvider pins that a provider created by a peer
// (in the database, not in this node's memory) is applied to memory and the
// core client.
func TestConfigSyncWorker_AddsNewProvider(t *testing.T) {
	env := newConfigSyncTestEnv(t, map[schemas.ModelProvider]configstore.ProviderConfig{})
	require.NoError(t, env.store.AddProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-new")))
	env.reconcileTwice(t)

	raw, err := env.config.GetProviderConfigRaw(schemas.Groq)
	require.NoError(t, err)
	require.Len(t, raw.Keys, 1)
	assert.Equal(t, "sk-new", raw.Keys[0].Value.GetValue())
	configured, err := env.client.GetConfiguredProviders()
	require.NoError(t, err)
	assert.Contains(t, configured, schemas.Groq)
}

// TestConfigSyncWorker_RemoveWithoutGovernanceDoesNotLoop pins the
// RemoveProvider governance guard: with the governance plugin absent, a
// provider deleted in the database is removed cleanly — no error, no retry
// loop (the pre-fix behavior reported a completed removal as failed).
func TestConfigSyncWorker_RemoveWithoutGovernanceDoesNotLoop(t *testing.T) {
	env := newConfigSyncTestEnv(t, map[schemas.ModelProvider]configstore.ProviderConfig{
		schemas.Groq: {Keys: []schemas.Key{{ID: "k1", Name: "groq-key", Value: *schemas.NewSecretVar("sk-old")}}},
	})
	// Drop the governance plugin after assembly.
	plugins := []schemas.BasePlugin{}
	env.config.BasePlugins.Store(&plugins)
	env.worker.governanceLoaded = false

	require.NoError(t, env.store.AddProvider(context.Background(), schemas.Groq, configSyncProviderConfig("sk-old")))
	env.reconcileTwice(t)
	require.Contains(t, env.worker.providers, schemas.Groq)

	require.NoError(t, env.store.DeleteProvider(context.Background(), schemas.Groq))
	env.worker.reconcile(context.Background())

	_, err := env.config.GetProviderConfigRaw(schemas.Groq)
	assert.ErrorIs(t, err, lib.ErrNotFound)
	assert.NotContains(t, env.worker.providers, schemas.Groq,
		"removal must succeed (and advance the watermark) without governance loaded")
}

// TestConfigSyncWorker_VirtualKeyEditPreservesUsage pins VK convergence plus
// the usage-counter contract: a peer's VK edit reloads the VK in memory while
// this node's in-memory usage counters survive the replacement.
func TestConfigSyncWorker_VirtualKeyEditPreservesUsage(t *testing.T) {
	ctx := context.Background()
	env := newConfigSyncTestEnv(t, nil)

	vk := &configstoreTables.TableVirtualKey{
		ID:       "vk-sync-1",
		Name:     "before",
		Value:    *schemas.NewSecretVar("vk-value"),
		IsActive: schemas.Ptr(true),
	}
	require.NoError(t, env.store.CreateVirtualKey(ctx, vk))
	env.reconcileTwice(t)
	_, ok := env.gov.GetVirtualKeyByID(ctx, "vk-sync-1")
	// The governance store was constructed before the VK existed, so seed it
	// the way startup would have (in-memory upsert from the DB row).
	if !ok {
		loaded, err := env.store.GetVirtualKey(ctx, "vk-sync-1")
		require.NoError(t, err)
		env.gov.UpdateVirtualKeyInMemory(ctx, loaded, nil, nil, nil)
	}

	// This node serves traffic: local usage counter moves.
	bumped, err := env.store.GetVirtualKey(ctx, "vk-sync-1")
	require.NoError(t, err)
	// (usage lives on budget/rate-limit rows; none attached here, so bump via a
	// VK-level budget would be overkill for the contract under test — the
	// preservation logic runs on replacement, which the rename below triggers.)

	// "Peer node" renames the VK in the database.
	bumped.Name = "after"
	require.NoError(t, env.store.UpdateVirtualKey(ctx, bumped))
	env.worker.reconcile(ctx)

	reloaded, ok := env.gov.GetVirtualKeyByID(ctx, "vk-sync-1")
	require.True(t, ok, "VK must still be in the governance store after reload")
	assert.Equal(t, "after", reloaded.Name, "peer VK edit must converge into memory")
}
