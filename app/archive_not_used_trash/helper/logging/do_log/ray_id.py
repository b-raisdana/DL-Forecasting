import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

ray_id_var: ContextVar[uuid.UUID | None] = ContextVar("ray_id")


# تابع برای ذخیره UUID در ContextVar


# تابع برای بازیابی UUID از ContextVar


class ContextVarMiddleware(BaseHTTPMiddleware):
    """
    This middleware should be registered as the last middleware so that
    it runs first in the request lifecycle.
    Register it in your FastAPI app like this:app.add_middleware(ContextVarMiddleware)
    """

    async def dispatch(self, request: Request, call_next):
        # Clean up or initialize the ContextVar at the start of a request
        ray_id_var.set(None)

        try:
            # Process the request
            response = await call_next(request)
        finally:
            # Clean up the ContextVar at the end of a request
            ray_id_var.set(None)

        return response
