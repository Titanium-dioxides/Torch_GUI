import axios from "axios";

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const message = err.response?.data?.message ?? err.message;
    const detail = err.response?.data?.detail;
    console.error(`[API Error] ${message}`, detail);
    console.error(`[Full Error]`, err.response?.data);
    return Promise.reject(new Error(message));
  }
);