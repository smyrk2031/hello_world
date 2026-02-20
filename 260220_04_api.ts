import axios from 'axios'

// VITE_API_BASE_URL が設定されていれば API はそこへ（別サーバ・別サブパス）。未設定なら同一オリジンで VITE_APP_BASE_PATH を基準に送る
const explicitApiUrl = (import.meta.env.VITE_API_BASE_URL || '').trim()
const appBase = import.meta.env.PROD ? (import.meta.env.VITE_APP_BASE_PATH || '').trim().replace(/\/+$/, '') : ''
const API_BASE_URL = explicitApiUrl || appBase || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true, // セッションCookieを送信
  headers: {
    'Content-Type': 'application/json',
  },
})

// リクエストインターセプター
api.interceptors.request.use(
  (config) => {
    // FormDataの場合はContent-Typeを自動設定（boundaryを含む）
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type']
    }
    // 必要に応じて認証トークンなどを追加
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// レスポンスインターセプター
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response?.status === 401) {
      // 認証エラーの処理
      console.error('認証が必要です')
    }
    return Promise.reject(error)
  }
)

export default api

