// API client helpers that wrap the generated OpenAPI SDK located in `api/index.ts`

import { AnswerRetrievalApi, Configuration, DefaultApi, type TopAnswersResponse } from "@/api"
import type { PredictionResponse } from "@/api"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const configuration = new Configuration({
  basePath: API_BASE_URL,
})

// Lazily instantiate clients so they can reuse the same configuration
const defaultApi = new DefaultApi(configuration)
const answerRetrievalApi = new AnswerRetrievalApi(configuration)

export async function predictCategory(query: string): Promise<PredictionResponse> {
  return defaultApi.predictPredictPost({
    predictionRequest: {
      text: query,
    },
  })
}

export async function findTopAnswers(query: string): Promise<TopAnswersResponse> {
  return answerRetrievalApi.findTopAnswersFindAnswerPost({
    answerRequest: {
      text: query,
    },
  })
}
