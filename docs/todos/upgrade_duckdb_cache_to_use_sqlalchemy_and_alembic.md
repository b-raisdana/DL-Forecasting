Here's a complete, worked version combining both — Pandera as source of truth, generating SQLAlchemy Core `Table`s, then imperatively mapping full ORM classes (with relationships) on top of them.

## The challenge with relationships

Pandera has no native concept of foreign keys or relationships — it validates DataFrame shape, not relational structure. So the generator handles columns/types/constraints from Pandera, and you specify FKs/relationships separately, alongside the schema. That's the one piece that can't be fully derived — everything else (columns, types, nullability, PK) comes from Pandera.

## 1. Define Pandera schemas (source of truth for columns)

```python
import pandera as pa
from pandera.typing import Series
from datetime import datetime
from typing import Optional

class CustomerSchema(pa.DataFrameModel):
    id: Series[int] = pa.Field(ge=1)
    name: Series[str]
    region: Optional[Series[str]] = pa.Field(nullable=True)

    class Config:
        coerce = True

class OrderSchema(pa.DataFrameModel):
    id: Series[int] = pa.Field(ge=1)
    customer_id: Series[int]
    amount: Series[float] = pa.Field(ge=0)
    created_at: Series[datetime]

    class Config:
        coerce = True
```

## 2. Generator: Pandera schema → SQLAlchemy Core `Table`

```python
import numpy as np
from sqlalchemy import Table, Column, MetaData, ForeignKey
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean, TypeEngine

PANDAS_TO_SQLA: dict[type, type[TypeEngine]] = {
    np.dtype("int64"): Integer,
    np.dtype("float64"): Float,
    np.dtype("bool"): Boolean,
    np.dtype("datetime64[ns]"): DateTime,
    np.dtype("object"): String,
}

def pandera_to_table(
    schema_cls: type[pa.DataFrameModel],
    table_name: str,
    metadata: MetaData,
    primary_key: str,
    foreign_keys: dict[str, str] | None = None,   # {"customer_id": "customers.id"}
) -> Table:
    schema = schema_cls.to_schema()
    foreign_keys = foreign_keys or {}
    columns = []

    for name, col in schema.columns.items():
        sqla_type = PANDAS_TO_SQLA.get(np.dtype(col.dtype.type), String)
        col_args = [name, sqla_type]
        if name in foreign_keys:
            col_args.append(ForeignKey(foreign_keys[name]))
        columns.append(
            Column(*col_args, primary_key=(name == primary_key), nullable=col.nullable)
        )

    return Table(table_name, metadata, *columns)
```

## 3. Generate tables + imperatively map ORM classes with relationships

```python
from sqlalchemy import MetaData
from sqlalchemy.orm import registry, relationship

metadata = MetaData()
mapper_registry = registry(metadata=metadata)

customers_table = pandera_to_table(CustomerSchema, "customers", metadata, primary_key="id")
orders_table = pandera_to_table(
    OrderSchema, "orders", metadata,
    primary_key="id",
    foreign_keys={"customer_id": "customers.id"},
)

class Customer:
    def __repr__(self):
        return f"Customer(id={self.id!r}, name={self.name!r}, region={self.region!r})"

class Order:
    def __repr__(self):
        return f"Order(id={self.id!r}, customer_id={self.customer_id!r}, amount={self.amount!r})"

mapper_registry.map_imperatively(
    Customer, customers_table,
    properties={"orders": relationship(Order, back_populates="customer")},
)
mapper_registry.map_imperatively(
    Order, orders_table,
    properties={"customer": relationship(Customer, back_populates="orders")},
)
```

At this point `Customer`/`Order` are fully normal ORM classes — same as declarative classes — but their columns came entirely from `CustomerSchema`/`OrderSchema`. One definition of "what an order looks like," reused for both DataFrame validation and DB mapping.

## 4. `db.py` — engine, unchanged from before

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

engine = create_engine("duckdb:///analytics.duckdb", poolclass=NullPool)
Session = sessionmaker(bind=engine)

metadata.create_all(engine)  # creates customers + orders using the generated Tables
```

## 5. CRUD — identical usage to a hand-written declarative model

```python
from db import Session

with Session() as session:
    session.add(Customer(id=1, name="Acme", region="EMEA"))
    session.add(Order(id=1, customer_id=1, amount=1200.0, created_at=datetime.utcnow()))
    session.commit()
```

## 6. Join + aggregation — exact same syntax as before

```python
from sqlalchemy import select, func
from db import Session

with Session() as session:
    stmt = (
        select(
            Customer.region,
            func.sum(Order.amount).label("total_revenue"),
            func.count(Order.id).label("order_count"),
        )
        .join(Order, Order.customer_id == Customer.id)
        .group_by(Customer.region)
        .order_by(func.sum(Order.amount).desc())
    )
    for row in session.execute(stmt):
        print(row.region, row.total_revenue, row.order_count)
```

## 7. Validate a DataFrame with Pandera, then `merge()` each row through the ORM

```python
import pandas as pd
from db import Session

df = pd.DataFrame({
    "id": [1, 2],
    "name": ["Acme Corp", "Globex"],
    "region": ["EMEA", "AMER"],
})
validated = CustomerSchema.validate(df)

with Session() as session:
    for row in validated.itertuples(index=False):
        session.merge(Customer(id=row.id, name=row.name, region=row.region))
    session.commit()
```

## 8. Validate + bulk upsert (SQL-level, for larger batches)

```python
from sqlalchemy import text
from db import Session

def upsert_orders(df: pd.DataFrame):
    validated = OrderSchema.validate(df)  # raises pa.errors.SchemaError on bad data
    with Session() as session:
        session.execute(
            text("""
                INSERT INTO orders (id, customer_id, amount, created_at)
                VALUES (:id, :customer_id, :amount, :created_at)
                ON CONFLICT (id) DO UPDATE SET
                    amount = excluded.amount,
                    created_at = excluded.created_at
            """),
            validated.to_dict(orient="records"),
        )
        session.commit()

upsert_orders(pd.DataFrame({
    "id": [1, 3],
    "customer_id": [1, 2],
    "amount": [1500.0, 300.0],
    "created_at": [datetime.utcnow()] * 2,
}))
```

Since `validated` was checked against `OrderSchema` — the same schema that generated the `orders` table's columns — there's no drift possible between what Pandera allows through and what the table accepts.

## 9. Alembic compatibility

Alembic autogenerate works against `target_metadata` — since `metadata` here is a real `MetaData` object populated by `pandera_to_table`, it plugs in exactly like a declarative `Base.metadata` would:

```python
# alembic/env.py
from your_module import metadata  # the same MetaData built from Pandera schemas
target_metadata = metadata
```

```bash
alembic revision --autogenerate -m "create customers and orders from pandera schemas"
alembic upgrade head
```

One caveat: if you change a Pandera schema (add a field, change a type) and regenerate the `Table`, you need to re-run the generator _before_ calling `alembic revision --autogenerate`, since Alembic diffs against whatever `metadata` currently holds — not against the Pandera class definitions directly. In practice this means: Pandera schema change → re-import/rebuild `metadata` → autogenerate → review the migration → apply. The generator is a build step, not something Alembic understands natively.

## What this buys you, concretely

|                                      | Before (double coding)                             | Now                                                                                                                                                                                                                              |
| ------------------------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Column names/types                   | Defined in both `DataFrameModel` and `Column(...)` | Defined once in `DataFrameModel`                                                                                                                                                                                                 |
| Field constraints (`ge=0`, nullable) | Pandera-only, not reflected in ORM                 | Nullability flows through; value constraints (`ge=0`) still Pandera-only — SQLAlchemy has no generic "check constraint from Pandera Field" bridge, so add `CheckConstraint` manually if you need it enforced at the DB level too |
| Relationships/FKs                    | Hand-written either way                            | Still hand-written (Pandera has no relationship concept) — the one irreducible piece                                                                                                                                             |
| Migration history                    | Alembic vs. hand-maintained models could drift     | Alembic diffs the _generated_ `MetaData`, same as any declarative setup                                                                                                                                                          |

The remaining manual work is genuinely irreducible (relationships, FK wiring, DB-level `CheckConstraint`s) — everything else that used to require touching two class definitions now only requires touching the Pandera model.
