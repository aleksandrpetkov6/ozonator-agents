import { app, BrowserWindow, ipcMain, nativeTheme, safeStorage, net, dialog } from 'electron'
import { join } from 'path'
import { appendFileSync, mkdirSync } from 'fs'
import { ensureDb, dbGetAdminSettings, dbSaveAdminSettings, dbIngestLifecycleMarkers, dbGetProducts, dbGetSyncLog, dbClearLogs, dbLogFinish, dbLogStart, dbUpsertProducts, dbDeleteProductsMissingForStore, dbCountProducts, dbReplaceProductPlacementsForStore, dbGetGridColumns, dbSaveGridColumns, dbRecordApiRawResponse } from './storage/db'
import { deleteSecrets, hasSecrets, loadSecrets, saveSecrets, updateStoreName } from './storage/secrets'
import { ozonGetStoreName, ozonPlacementZoneInfo, ozonProductInfoList, ozonProductList, ozonTestAuth, ozonWarehouseList, setOzonApiCaptureHook } from './ozon'
import { type SalesPeriod } from './sales-sync'
import { ensureLocalSalesSnapshotFromApiIfMissing, getLocalDatasetRows, refreshCoreLocalDatasetSnapshots, refreshSalesRawSnapshotFromApi } from './local-datasets'
let mainWindow: BrowserWindow | null = null
let startupShowTimer: NodeJS.Timeout | null = null
let backgroundSyncTimer: NodeJS.Timeout | null = null
let isQuitting = false
let syncProductsInFlight: Promise<any> | null = null

const singleInstanceLock = app.requestSingleInstanceLock()
if (!singleInstanceLock) {
app.quit()
}
function startupLog(...args: any[]) {
try {
const dir = app?.isReady?.() ? app.getPath('userData') : app.getPath('temp')
mkdirSync(dir, { recursive: true })
const line = `[${new Date().toISOString()}] ` + args.map((a) => {
try { return typeof a === 'string' ? a : JSON.stringify(a) } catch { return String(a) }
}).join(' ') + '\n'
appendFileSync(join(dir, 'ozonator-startup.log'), line, 'utf8')
} catch {}
try { console.log('[startup]', ...args) } catch {}
}
function safeShowMainWindow(reason: string) {
try {
if (!mainWindow || mainWindow.isDestroyed()) return
startupLog('safeShowMainWindow', { reason, visible: mainWindow.isVisible() })
if (!mainWindow.isVisible()) {
try { mainWindow.show() } catch {}
}
try { mainWindow.focus() } catch {}
try { mainWindow.maximize() } catch {}
} catch (e: any) {
startupLog('safeShowMainWindow.error', e?.message ?? String(e))
}
}
function chunk<T>(arr: T[], size: number): T[][] {
const out: T[][] = []
for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
return out
}
function createWindow() {
startupLog('createWindow.begin', { packaged: app.isPackaged, appPath: app.getAppPath(), __dirname })
mainWindow = new BrowserWindow({
width: 1200,
height: 760,
minWidth: 980,
minHeight: 620,
title: 'Озонатор',
show: false,
backgroundColor: '#F5F5F7',
autoHideMenuBar: true,
titleBarOverlay: { color: '#F5F5F7', symbolColor: '#1d1d1f', height: 34 },
webPreferences: {
preload: join(__dirname, '../preload/index.js'),
contextIsolation: true,
nodeIntegration: false,
backgroundThrottling: false,
},
})
if (startupShowTimer) {
clearTimeout(startupShowTimer)
startupShowTimer = null
}
startupShowTimer = setTimeout(() => safeShowMainWindow('show-timeout-fallback'), 2500)
mainWindow.once('ready-to-show', () => {
startupLog('event.ready-to-show')
safeShowMainWindow('ready-to-show')
})
mainWindow.webContents.on('did-finish-load', () => {
startupLog('event.did-finish-load', { url: mainWindow?.webContents?.getURL?.() })
safeShowMainWindow('did-finish-load')
})
mainWindow.webContents.on('did-fail-load', (_e, code, desc, url, isMainFrame) => {
startupLog('event.did-fail-load', { code, desc, url, isMainFrame })
try {
if (isMainFrame && mainWindow && !mainWindow.isDestroyed()) {
        const html = `<!doctype html><html><body style="font-family:Segoe UI,sans-serif;padding:16px">
          <h3>Озонатор не смог загрузить интерфейс</h3>
          <div>Причина: ${String(desc || 'did-fail-load')} (code ${String(code)})</div>
          <div style="margin-top:8px;color:#555">Подробности в файле ozonator-startup.log в папке данных приложения.</div>
        </body></html>`
mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {})
}
} catch {}
safeShowMainWindow('did-fail-load')
})
mainWindow.webContents.on('render-process-gone', (_e, details) => {
startupLog('event.render-process-gone', details)
safeShowMainWindow('render-process-gone')
})
mainWindow.on('unresponsive', () => {
startupLog('event.window-unresponsive')
})
mainWindow.on('close', (event) => {
if (isQuitting) return
event.preventDefault()
startupLog('event.window-close-hide')
try { mainWindow?.hide() } catch {}
void runBackgroundSyncTick('window-close')
})
mainWindow.on('closed', () => {
startupLog('event.window-closed')
if (startupShowTimer) {
clearTimeout(startupShowTimer)
startupShowTimer = null
}
mainWindow = null
})
const devUrl = process.env.ELECTRON_RENDERER_URL || process.env.VITE_DEV_SERVER_URL || (!app.isPackaged ? 'http://localhost:5173/' : null)
startupLog('renderer.target', { devUrl, packaged: app.isPackaged })
if (devUrl) {
mainWindow.loadURL(devUrl).catch((e) => startupLog('loadURL.error', e?.message ?? String(e)))
try { mainWindow.webContents.openDevTools({ mode: 'detach' }) } catch {}
} else {
const rendererFile = join(app.getAppPath(), 'out/renderer/index.html')
startupLog('renderer.file', rendererFile)
mainWindow.loadFile(rendererFile).catch((e) => startupLog('loadFile.error', e?.message ?? String(e)))
}
nativeTheme.themeSource = 'light'
}
app.whenReady().then(() => {
try {
startupLog('app.whenReady')
if (!safeStorage.isEncryptionAvailable()) {
console.warn('safeStorage encryption is not available on this machine.')
startupLog('safeStorage.unavailable')
}
ensureDb()
startupLog('ensureDb.ok')
setOzonApiCaptureHook((evt) => {
dbRecordApiRawResponse({
storeClientId: evt.storeClientId,
method: evt.method,
endpoint: evt.endpoint,
requestBody: evt.requestBody,
responseBody: evt.responseBody,
httpStatus: evt.httpStatus,
isSuccess: evt.isSuccess,
errorMessage: evt.errorMessage ?? null,
fetchedAt: evt.fetchedAt,
})
})
dbIngestLifecycleMarkers({ appVersion: app.getVersion() })
startupLog('dbIngestLifecycleMarkers.ok', { version: app.getVersion() })
createWindow()
ensureBackgroundSyncLoop()
app.on('second-instance', () => {
startupLog('app.second-instance')
safeShowMainWindow('second-instance')
})
app.on('before-quit', () => {
isQuitting = true
if (backgroundSyncTimer) {
clearInterval(backgroundSyncTimer)
backgroundSyncTimer = null
}
})
app.on('activate', () => {
startupLog('app.activate', { windows: BrowserWindow.getAllWindows().length })
if (BrowserWindow.getAllWindows().length === 0) createWindow()
else safeShowMainWindow('app-activate')
})
} catch (e: any) {
startupLog('fatal.startup', e?.stack ?? e?.message ?? String(e))
try {
dialog.showErrorBox('Озонатор — ошибка запуска', String(e?.message ?? e))
} catch {}
try {
if (!mainWindow) {
mainWindow = new BrowserWindow({ width: 900, height: 640, show: true, autoHideMenuBar: true })
        const html = `<!doctype html><html><body style="font-family:Segoe UI,sans-serif;padding:16px">
          <h3>Озонатор не запустился</h3>
          <pre style="white-space:pre-wrap">${String(e?.stack ?? e?.message ?? e)}</pre>
          <div style="color:#555">Подробности: ozonator-startup.log в папке данных приложения.</div>
        </body></html>`
mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {})
}
} catch {}
}
})
process.on('uncaughtException', (e: any) => {
startupLog('process.uncaughtException', e?.stack ?? e?.message ?? String(e))
})
process.on('unhandledRejection', (e: any) => {
startupLog('process.unhandledRejection', e as any)
})
app.on('window-all-closed', () => {
if (process.platform !== 'darwin') app.quit()
})
function checkInternet(timeoutMs = 2500): Promise<boolean> {
return new Promise((resolve) => {
const request = net.request({ method: 'GET', url: 'https://api-seller.ozon.ru' })
const timer = setTimeout(() => {
try { request.abort() } catch {}
resolve(false)
}, timeoutMs)
request.on('response', () => {
clearTimeout(timer)
resolve(true)
})
request.on('error', () => {
clearTimeout(timer)
resolve(false)
})
request.end()
})
}
function getActiveStoreClientIdSafe(): string | null {
try {
return loadSecrets().clientId
} catch {
return null
}
}

function readDatasetRowsSafe(datasetRaw: string, period?: SalesPeriod | null) {
const storeClientId = getActiveStoreClientIdSafe()
const dataset = String(datasetRaw ?? '').trim() || 'products'
const rows = getLocalDatasetRows(storeClientId, dataset, { period: period ?? null })
return { storeClientId, dataset, rows }
}



function emitRendererDataUpdatedEvents() {
if (!mainWindow || mainWindow.isDestroyed()) return
const script = `
try {
window.dispatchEvent(new Event('ozon:products-updated'))
window.dispatchEvent(new Event('ozon:logs-updated'))
window.dispatchEvent(new Event('ozon:store-updated'))
} catch {}
`
try {
void mainWindow.webContents.executeJavaScript(script, true)
} catch {}
}

async function performProductsSync(args?: { salesPeriod?: SalesPeriod | null }) {
if (syncProductsInFlight) return await syncProductsInFlight
const job = (async () => {
let storeClientId: string | null = null
try { storeClientId = loadSecrets().clientId } catch {}
const logId = dbLogStart('sync_products', storeClientId)
try {
const secrets = loadSecrets()
const existingOfferIds = new Set(dbGetProducts(secrets.clientId).map((p: any) => p.offer_id))
const incomingOfferIds = new Set<string>()
let added = 0
let lastId = ''
const limit = 1000
let pages = 0
let total = 0
for (let guard = 0; guard < 200; guard++) {
const { items, lastId: next, total: totalMaybe } = await ozonProductList(secrets, { lastId, limit })
pages += 1
total += items.length
const ids = items.map(i => i.product_id).filter(Boolean) as number[]
const infoList = await ozonProductInfoList(secrets, ids)
const infoMap = new Map<number, typeof infoList[number]>()
for (const p of infoList) infoMap.set(p.product_id, p)
const enriched = items.map((it) => {
const info = it.product_id ? infoMap.get(it.product_id) : undefined
return {
offer_id: it.offer_id,
product_id: it.product_id,
sku: (info?.ozon_sku ?? info?.sku ?? it.sku ?? null),
ozon_sku: (info?.ozon_sku ?? info?.sku ?? it.sku ?? null),
seller_sku: (info?.seller_sku ?? it.offer_id ?? null),
fbo_sku: info?.fbo_sku ?? null,
fbs_sku: info?.fbs_sku ?? null,
barcode: info?.barcode ?? null,
brand: info?.brand ?? null,
category: info?.category ?? null,
type: info?.type ?? null,
name: info?.name ?? null,
photo_url: info?.photo_url ?? null,
is_visible: info?.is_visible ?? null,
hidden_reasons: info?.hidden_reasons ?? null,
created_at: info?.created_at ?? null,
archived: it.archived ?? false,
store_client_id: secrets.clientId,
}
})
for (const it of enriched) {
const offer = String((it as any).offer_id)
if (offer) incomingOfferIds.add(offer)
if (!existingOfferIds.has(offer)) {
existingOfferIds.add(offer)
added += 1
}
}
dbUpsertProducts(enriched)
if (!next) break
if (next === lastId) break
lastId = next
if (typeof totalMaybe === 'number' && total >= totalMaybe) break
}
dbDeleteProductsMissingForStore(secrets.clientId, Array.from(incomingOfferIds))
const syncedCount = dbCountProducts(secrets.clientId)
let placementRowsCount = 0
let placementSyncError: string | null = null
let placementCacheKept = false
try {
const productsForStore = dbGetProducts(secrets.clientId)
const ozonSkuList = Array.from(new Set(productsForStore.map((p) => String(p.sku ?? '').trim()).filter(Boolean)))
const sellerSkuList = Array.from(new Set(productsForStore.map((p) => String(p.offer_id ?? '').trim()).filter(Boolean)))
if (ozonSkuList.length > 0 || sellerSkuList.length > 0) {
const warehouses = await ozonWarehouseList(secrets)
if (!Array.isArray(warehouses) || warehouses.length === 0) {
placementSyncError = 'Ozon не вернул список складов; локальные данные по складам/зонам сохранены без перезаписи.'
placementCacheKept = true
} else {
const allPlacementRows: Array<{
warehouse_id: number
warehouse_name?: string | null
sku: string
ozon_sku?: string | null
seller_sku?: string | null
placement_zone?: string | null
}> = []
const placementRowKeys = new Set<string>()
let placementApiCallCount = 0
const appendPlacementRows = (
warehouseId: number,
warehouseName: string | null,
zones: Array<{
sku: string
ozon_sku?: string | null
seller_sku?: string | null
placement_zone: string | null
}>
) => {
for (const z of zones) {
const rowKey = [
String(warehouseId),
String(z.ozon_sku ?? ''),
String(z.seller_sku ?? ''),
String(z.placement_zone ?? ''),
].join('::')
if (placementRowKeys.has(rowKey)) continue
placementRowKeys.add(rowKey)
allPlacementRows.push({
warehouse_id: warehouseId,
warehouse_name: warehouseName,
sku: z.sku,
ozon_sku: z.ozon_sku ?? null,
seller_sku: z.seller_sku ?? null,
placement_zone: z.placement_zone ?? null,
})
}
}
for (const wh of warehouses) {
const wid = Number(wh.warehouse_id)
if (!Number.isFinite(wid)) continue
for (const part of chunk(ozonSkuList, 500)) {
placementApiCallCount += 1
const zones = await ozonPlacementZoneInfo(secrets, { warehouseId: wid, skus: part })
appendPlacementRows(wid, wh.name ?? null, zones)
}
for (const part of chunk(sellerSkuList, 500)) {
placementApiCallCount += 1
const zones = await ozonPlacementZoneInfo(secrets, { warehouseId: wid, skus: part })
appendPlacementRows(wid, wh.name ?? null, zones)
}
}
if (allPlacementRows.length === 0 && placementApiCallCount > 0) {
placementSyncError = 'Ozon не вернул зоны размещения ни по одному SKU; прежние локальные данные по складам/зонам сохранены.'
placementCacheKept = true
} else {
placementRowsCount = dbReplaceProductPlacementsForStore(secrets.clientId, allPlacementRows)
}
}
} else {
placementRowsCount = dbReplaceProductPlacementsForStore(secrets.clientId, [])
}
} catch (placementErr: any) {
placementSyncError = placementErr?.message ?? String(placementErr)
}
const localSnapshots = refreshCoreLocalDatasetSnapshots(secrets.clientId)
let salesRowsCount = 0
let salesSyncError: string | null = null
try {
const salesRefresh = await refreshSalesRawSnapshotFromApi(secrets, args?.salesPeriod ?? null)
salesRowsCount = Number(salesRefresh?.rowsCount ?? 0)
} catch (salesErr: any) {
salesSyncError = salesErr?.message ?? String(salesErr)
}
if (!secrets.storeName) {
try {
const name = await ozonGetStoreName(secrets)
if (name) updateStoreName(name)
} catch {
}
}
dbLogFinish(logId, {
status: 'success',
itemsCount: syncedCount,
storeClientId: secrets.clientId,
meta: {
added,
storeClientId: secrets.clientId,
storeName: loadSecrets().storeName ?? null,
placementRowsCount,
placementSyncError,
placementCacheKept,
localProductsRowsCount: localSnapshots.productsRowsCount,
localStocksRowsCount: localSnapshots.stocksRowsCount,
salesRowsCount,
salesSyncError,
},
})
return { ok: true, itemsCount: syncedCount, pages, placementRowsCount, placementSyncError, salesRowsCount, salesSyncError }
} catch (e: any) {
dbLogFinish(logId, { status: 'error', errorMessage: e?.message ?? String(e), errorDetails: e?.details, storeClientId })
return { ok: false, error: e?.message ?? String(e) }
}
})()
syncProductsInFlight = job
try {
return await job
} finally {
if (syncProductsInFlight === job) syncProductsInFlight = null
}
}

async function runBackgroundSyncTick(reason: string) {
if (isQuitting) return
if (!mainWindow || mainWindow.isDestroyed()) return
if (!hasSecrets()) return
const online = await checkInternet()
if (!online) return
const resp = await performProductsSync({ salesPeriod: null })
if (resp?.ok) {
startupLog('background-sync.ok', {
reason,
itemsCount: Number(resp?.itemsCount ?? 0),
windowVisible: mainWindow.isVisible(),
})
emitRendererDataUpdatedEvents()
} else if (resp?.error) {
startupLog('background-sync.error', { reason, error: String(resp.error) })
}
}

function ensureBackgroundSyncLoop() {
if (backgroundSyncTimer) {
clearInterval(backgroundSyncTimer)
backgroundSyncTimer = null
}
backgroundSyncTimer = setInterval(() => {
void runBackgroundSyncTick('interval')
}, 60 * 1000)
}

ipcMain.handle('secrets:status', async () => {
return {
hasSecrets: hasSecrets(),
encryptionAvailable: safeStorage.isEncryptionAvailable(),
}
})
ipcMain.handle('secrets:save', async (_e, secrets: { clientId: string; apiKey: string }) => {
saveSecrets({ clientId: String(secrets.clientId).trim(), apiKey: String(secrets.apiKey).trim() })
return { ok: true }
})
ipcMain.handle('secrets:load', async () => {
const s = loadSecrets()
return { ok: true, secrets: { clientId: s.clientId, apiKey: s.apiKey, storeName: s.storeName ?? null } }
})
ipcMain.handle('secrets:delete', async () => {
deleteSecrets()
return { ok: true }
})
ipcMain.handle('net:check', async () => {
return { online: await checkInternet() }
})
ipcMain.handle('admin:getSettings', async () => {
try {
return { ok: true, ...dbGetAdminSettings() }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), logRetentionDays: 30 }
}
})
ipcMain.handle('admin:saveSettings', async (_e, payload: { logRetentionDays?: number }) => {
try {
const saved = dbSaveAdminSettings({ logRetentionDays: Number(payload?.logRetentionDays) })
return { ok: true, ...saved }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e) }
}
})
ipcMain.handle('ozon:testAuth', async () => {
let storeClientId: string | null = null
try { storeClientId = loadSecrets().clientId } catch {}
const logId = dbLogStart('check_auth', storeClientId)
try {
const secrets = loadSecrets()
await ozonTestAuth(secrets)
try {
const name = await ozonGetStoreName(secrets)
if (name) updateStoreName(name)
} catch {
}
dbLogFinish(logId, { status: 'success', storeClientId: secrets.clientId })
const refreshed = loadSecrets()
return { ok: true, storeName: refreshed.storeName ?? null }
} catch (e: any) {
dbLogFinish(logId, { status: 'error', errorMessage: e?.message ?? String(e), errorDetails: e?.details, storeClientId })
return { ok: false, error: e?.message ?? String(e) }
}
})
ipcMain.handle('ozon:syncProducts', async (_e, args?: { salesPeriod?: SalesPeriod | null }) => {
const resp = await performProductsSync(args)
if (resp?.ok) emitRendererDataUpdatedEvents()
return resp
})
ipcMain.handle('data:refreshSales', async (_e, args?: { period?: SalesPeriod | null }) => {
try {
const secrets = loadSecrets()
const refreshed = await refreshSalesRawSnapshotFromApi(secrets, args?.period ?? null)
return { ok: true, rowsCount: Number(refreshed?.rowsCount ?? 0), rateLimited: false }
} catch (e: any) {
const message = e?.message ?? String(e)
const isRateLimited = /HTTP\s*429/.test(message)
if (isRateLimited) {
try {
const rows = getLocalDatasetRows(getActiveStoreClientIdSafe(), 'sales', { period: args?.period ?? null })
if (Array.isArray(rows) && rows.length > 0) {
return { ok: true, rowsCount: rows.length, rateLimited: true }
}
} catch {
}
}
return { ok: false, error: message, rowsCount: 0, rateLimited: isRateLimited }
}
})
ipcMain.handle('data:getDatasetRows', async (_e, args?: { dataset?: string; period?: SalesPeriod | null }) => {
try {
const dataset = String(args?.dataset ?? 'products').trim() || 'products'
if (dataset === 'sales') {
let secrets = null
try {
secrets = loadSecrets()
} catch {
secrets = null
}
await ensureLocalSalesSnapshotFromApiIfMissing(secrets, args?.period ?? null)
const rows = getLocalDatasetRows(getActiveStoreClientIdSafe(), 'sales', { period: args?.period ?? null })
return { ok: true, dataset, rows }
}
const { rows } = readDatasetRowsSafe(dataset, args?.period ?? null)
return { ok: true, dataset, rows }
} catch (e: any) {
const dataset = String(args?.dataset ?? 'products').trim() || 'products'
return { ok: false, error: e?.message ?? String(e), dataset, rows: [] }
}
})
ipcMain.handle('data:getProducts', async () => {
try {
const { rows } = readDatasetRowsSafe('products', null)
return { ok: true, products: rows }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), products: [] }
}
})
ipcMain.handle('data:getSales', async (_e, args?: { period?: SalesPeriod | null }) => {
try {
let secrets = null
try {
secrets = loadSecrets()
} catch {
secrets = null
}
await ensureLocalSalesSnapshotFromApiIfMissing(secrets, args?.period ?? null)
const rows = getLocalDatasetRows(getActiveStoreClientIdSafe(), 'sales', { period: args?.period ?? null })
return { ok: true, rows }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), rows: [] }
}
})
ipcMain.handle('data:getReturns', async () => {
try {
const { rows } = readDatasetRowsSafe('returns', null)
return { ok: true, rows }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), rows: [] }
}
})
ipcMain.handle('data:getStocks', async () => {
try {
const { rows } = readDatasetRowsSafe('stocks', null)
return { ok: true, rows }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), rows: [] }
}
})
ipcMain.handle('ui:getGridColumns', async (_e, args: { dataset: string }) => {
try {
return { ok: true, ...dbGetGridColumns(args?.dataset) }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), dataset: String(args?.dataset ?? 'products'), cols: null }
}
})
ipcMain.handle('ui:saveGridColumns', async (_e, args: { dataset: string; cols: Array<{ id: string; w: number; visible: boolean; hiddenBucket: 'main' | 'add' }> }) => {
try {
return { ok: true, ...dbSaveGridColumns(args?.dataset, args?.cols) }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), dataset: String(args?.dataset ?? 'products'), savedCount: 0 }
}
})
ipcMain.handle('data:getSyncLog', async () => {
try {
let storeClientId: string | null = null
try {
storeClientId = loadSecrets().clientId
} catch {
storeClientId = null
}
const logs = dbGetSyncLog(storeClientId)
return { ok: true, logs }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e), logs: [] }
}
})
ipcMain.handle('data:clearLogs', async () => {
try {
dbClearLogs()
return { ok: true }
} catch (e: any) {
return { ok: false, error: e?.message ?? String(e) }
}
})
