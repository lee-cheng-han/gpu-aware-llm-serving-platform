from serving_platform.domain import RequestRecord, RequestState


class RequestLifecycle:
    """Small state-machine service used by control-plane and storage adapters."""

    def transition(
        self,
        request: RequestRecord,
        target: RequestState,
        at: float | None = None,
    ) -> RequestRecord:
        request.transition(target, at)
        return request
