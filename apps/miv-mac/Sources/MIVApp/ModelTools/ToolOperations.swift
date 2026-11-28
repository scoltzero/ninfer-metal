import Foundation

private enum ModelToolsAPI {
    static let hfTasks         = "/hf/tasks"
    static let hfDownload      = "/hf/download"
    static func hfCancel(_ taskId: String) -> String { "/hf/cancel/\(taskId)" }
    static func hfRetry(_ taskId: String) -> String  { "/hf/retry/\(taskId)" }
    static func hfTask(_ taskId: String) -> String   { "/hf/task/\(taskId)" }
    static let hfRecommended   = "/hf/recommended"
    static let hfSearch        = "/hf/search"
    static let hfModelInfo     = "/hf/model-info"
    static let msStatus        = "/ms/status"
    static let msTasks         = "/ms/tasks"
    static let msDownload      = "/ms/download"
    static func msCancel(_ taskId: String) -> String { "/ms/cancel/\(taskId)" }
    static func msRetry(_ taskId: String) -> String  { "/ms/retry/\(taskId)" }
    static func msTask(_ taskId: String) -> String   { "/ms/task/\(taskId)" }
    static let msRecommended   = "/ms/recommended"
    static let msSearch        = "/ms/search"
    static let msModelInfo     = "/ms/model-info"
    static let oqModels        = "/oq/models"
    static let oqEstimate      = "/oq/estimate"
    static let oqStart         = "/oq/start"
    static let oqTasks         = "/oq/tasks"
    static func oqCancel(_ taskId: String) -> String { "/oq/cancel/\(taskId)" }
    static func oqTask(_ taskId: String) -> String   { "/oq/task/\(taskId)" }
    static let uploadValidateToken = "/upload/validate-token"
    static let uploadModels        = "/upload/oq-models"
    static let uploadStart         = "/upload/start"
    static let uploadTasks         = "/upload/tasks"
    static func uploadCancel(_ taskId: String) -> String { "/upload/cancel/\(taskId)" }
    static func uploadTask(_ taskId: String) -> String   { "/upload/task/\(taskId)" }
}

extension ModelToolsClient {
    func listHFTasks() async throws -> HFTaskListResponse {
        try await get(ModelToolsAPI.hfTasks)
    }

    func startHFDownload(
        repoId: String,
        hfToken: String = "",
        files: [String]? = nil
    ) async throws -> StartHFDownloadResponse {
        try await post(ModelToolsAPI.hfDownload, body: StartHFDownloadRequest(
            repoId: repoId, hfToken: hfToken, files: files
        ))
    }

    func cancelHFDownload(taskId: String) async throws -> ToolStatus {
        try await postEmpty(ModelToolsAPI.hfCancel(taskId))
    }

    func retryHFDownload(taskId: String) async throws -> StartHFDownloadResponse {
        try await postEmpty(ModelToolsAPI.hfRetry(taskId))
    }

    func removeHFTask(taskId: String) async throws -> ToolStatus {
        try await delete(ModelToolsAPI.hfTask(taskId))
    }

    func getHFRecommended(modelFormat: HFModelFormat = .mlx) async throws -> HFRecommendedResponse {
        try await get(ModelToolsAPI.hfRecommended, query: [
            URLQueryItem(name: "model_format", value: modelFormat.rawValue),
        ])
    }

    func searchHFModels(
        query: String,
        sort: String = "trending",
        limit: Int = 20,
        modelFormat: HFModelFormat = .mlx
    ) async throws -> HFSearchResponse {
        try await get(ModelToolsAPI.hfSearch, query: [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "model_format", value: modelFormat.rawValue),
        ])
    }

    func getHFModelCard(repoId: String) async throws -> ModelCardDTO {
        try await get(ModelToolsAPI.hfModelInfo, query: [
            URLQueryItem(name: "repo_id", value: repoId),
        ])
    }

    func getMSStatus() async throws -> MSStatusResponse {
        try await get(ModelToolsAPI.msStatus)
    }

    func listMSTasks() async throws -> MSTaskListResponse {
        try await get(ModelToolsAPI.msTasks)
    }

    func startMSDownload(modelId: String, msToken: String = "") async throws -> StartMSDownloadResponse {
        try await post(ModelToolsAPI.msDownload, body: StartMSDownloadRequest(
            modelId: modelId, msToken: msToken
        ))
    }

    func cancelMSDownload(taskId: String) async throws -> ToolStatus {
        try await postEmpty(ModelToolsAPI.msCancel(taskId))
    }

    func retryMSDownload(taskId: String) async throws -> StartMSDownloadResponse {
        try await postEmpty(ModelToolsAPI.msRetry(taskId))
    }

    func removeMSTask(taskId: String) async throws -> ToolStatus {
        try await delete(ModelToolsAPI.msTask(taskId))
    }

    func getMSRecommended(mlxOnly: Bool = true) async throws -> MSRecommendedResponse {
        try await get(ModelToolsAPI.msRecommended, query: [
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    func searchMSModels(
        query: String,
        sort: String = "trending",
        limit: Int = 20,
        mlxOnly: Bool = true
    ) async throws -> MSSearchResponse {
        try await get(ModelToolsAPI.msSearch, query: [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "mlx_only", value: mlxOnly ? "true" : "false"),
        ])
    }

    func getMSModelCard(modelId: String) async throws -> ModelCardDTO {
        try await get(ModelToolsAPI.msModelInfo, query: [
            URLQueryItem(name: "model_id", value: modelId),
        ])
    }

    func listOQModels() async throws -> OQModelsResponse {
        try await get(ModelToolsAPI.oqModels)
    }

    func estimateOQ(
        modelPath: String,
        oqLevel: Double,
        preserveMtp: Bool = false,
        preserveNgramTable: Bool = false,
        textOnly: Bool = false,
        dtype: String = "bfloat16"
    ) async throws -> OQEstimateResponse {
        // `oq_level` accepts ints and fractional levels. Send it without a
        // trailing `.0` so the server parses an int when the user picked one.
        let levelStr: String = (oqLevel.rounded() == oqLevel)
            ? String(Int(oqLevel))
            : String(oqLevel)
        return try await get(ModelToolsAPI.oqEstimate, query: [
            URLQueryItem(name: "model_path", value: modelPath),
            URLQueryItem(name: "oq_level", value: levelStr),
            URLQueryItem(name: "preserve_mtp", value: preserveMtp ? "true" : "false"),
            URLQueryItem(name: "preserve_ngram_table", value: preserveNgramTable ? "true" : "false"),
            URLQueryItem(name: "text_only", value: textOnly ? "true" : "false"),
            URLQueryItem(name: "dtype", value: dtype),
        ])
    }

    func startOQQuantization(_ body: OQStartRequest) async throws -> OQStartResponse {
        try await post(ModelToolsAPI.oqStart, body: body)
    }

    func listOQTasks() async throws -> OQTasksResponse {
        try await get(ModelToolsAPI.oqTasks)
    }

    func cancelOQTask(taskId: String) async throws -> ToolStatus {
        try await postEmpty(ModelToolsAPI.oqCancel(taskId))
    }

    func removeOQTask(taskId: String) async throws -> ToolStatus {
        try await delete(ModelToolsAPI.oqTask(taskId))
    }

    func validateHFUploadToken(hfToken: String) async throws -> HFValidateTokenResponse {
        try await post(ModelToolsAPI.uploadValidateToken, body: HFValidateTokenRequest(hfToken: hfToken))
    }

    func listHFUploadModels() async throws -> HFUploadModelsResponse {
        try await get(ModelToolsAPI.uploadModels)
    }

    func startHFUpload(_ body: HFUploadStartRequest) async throws -> HFUploadStartResponse {
        try await post(ModelToolsAPI.uploadStart, body: body)
    }

    func listHFUploadTasks() async throws -> HFUploadTasksResponse {
        try await get(ModelToolsAPI.uploadTasks)
    }

    func cancelHFUploadTask(taskId: String) async throws -> ToolStatus {
        try await postEmpty(ModelToolsAPI.uploadCancel(taskId))
    }

    func removeHFUploadTask(taskId: String) async throws -> ToolStatus {
        try await delete(ModelToolsAPI.uploadTask(taskId))
    }
}
