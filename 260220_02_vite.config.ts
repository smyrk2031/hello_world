import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
// サブパス: 本番ビルド時のみ .env.production の VITE_APP_BASE_PATH を使用。開発時は常に base '/' でプロキシが動くようにする
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const appBaseRaw = (env.VITE_APP_BASE_PATH || process.env.VITE_APP_BASE_PATH || '').trim().replace(/\/+$/, '')
  const appBase = mode === 'production' ? appBaseRaw : ''
  const base = appBase ? `${appBase}/` : '/'

  return {
  base,
  
  plugins: [react()],
  
  // ビルド設定
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false, // 本番環境ではsourcemapを無効化（必要に応じて変更）
    rollupOptions: {
      output: {
        manualChunks: {
          // 大きなライブラリを分割してロード時間を最適化
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'chart-vendor': ['chart.js', 'react-chartjs-2'],
          'diagram-vendor': ['reactflow', 'three'],
        },
      },
    },
  },
  
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/auth': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/static': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      // /admin, /dashboard, /cms はReact Routerで処理するため、プロキシしない
      // /mypage, /app はReact Routerで処理するため、プロキシしない
      '/mycontents': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      // '/app' はReact Routerで処理するため、プロキシしない
      '/gallery': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/user': {
        target: env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  
  // プレビューサーバー設定（NSSM経由で本番環境で使用）
  preview: {
    port: 50000,
    strictPort: true, // ポートが使用中の場合はエラーにする
    host: '127.0.0.1', // localhostのみでリッスン
  },
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  }
})

