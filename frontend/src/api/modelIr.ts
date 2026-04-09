import { ModelIR } from "@/types/ir";
import { apiClient } from "./client";

export const modelIrApi = {
  save:    (ir: ModelIR)       => apiClient.post("/model-irs", ir),
  list:    ()                  => apiClient.get("/model-irs"),
  get:     (id: string)        => apiClient.get(`/model-irs/${id}`),
  update:  (id: string, ir: ModelIR) => apiClient.put(`/model-irs/${id}`, ir),
  delete:  (id: string)        => apiClient.delete(`/model-irs/${id}`),
  codegen: (id: string)        => apiClient.get<string>(`/model-irs/${id}/codegen`),
};