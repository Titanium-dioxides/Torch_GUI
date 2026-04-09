import { apiClient } from "./client";

export const experimentApi = {
  create: (ir: unknown)       => apiClient.post("/experiments", ir),
  submit: (expId: string)     => apiClient.post(`/experiments/${expId}/submit`),
  status: (expId: string)     => apiClient.get(`/experiments/${expId}/status`),
  cancel: (expId: string)     => apiClient.post(`/experiments/${expId}/cancel`),
};