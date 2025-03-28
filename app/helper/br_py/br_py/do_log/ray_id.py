import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

ray_id_var: ContextVar[uuid.UUID | None] = ContextVar('ray_id')


# تابع برای ذخیره UUID در ContextVar
def set_ray_id(id: uuid.UUID):
    ray_id_var.set(id)


# تابع برای بازیابی UUID از ContextVar
def get_ray_id(generate: bool = True) -> uuid.UUID:
    id_of_ray = ray_id_var.get(None)
    if generate and id_of_ray is None:
            id_of_ray = ray_id(source_type = 0)
            set_ray_id(id_of_ray)
    return id_of_ray


def ray_id(source_type: int, id_of_ray: uuid.UUID = None) -> uuid.UUID:
    """
    Define source_types inside the importer code. For example:
    Allocate left 3 octets to category
    'user_request': 0x00100001,
    'scheduled_type_a': 0x00200001,
    """
    if id_of_ray is not None:
        return id_of_ray
    id_of_ray = uuid.uuid1(node=source_type)
    return id_of_ray

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
