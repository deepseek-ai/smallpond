import importlib
import os.path
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import duckdb
import duckdb.typing


class UDFType(Enum):
    """
    A wrapper of duckdb.typing.DuckDBPyType

    See https://duckdb.org/docs/api/python/types.html
    """

    SQLNULL = 1
    BOOLEAN = 2
    TINYINT = 3
    UTINYINT = 4
    SMALLINT = 5
    USMALLINT = 6
    INTEGER = 7
    UINTEGER = 8
    BIGINT = 9
    UBIGINT = 10
    HUGEINT = 11
    UUID = 12
    FLOAT = 13
    DOUBLE = 14
    DATE = 15
    TIMESTAMP = 16
    TIMESTAMP_MS = 17
    TIMESTAMP_NS = 18
    TIMESTAMP_S = 19
    TIME = 20
    TIME_TZ = 21
    TIMESTAMP_TZ = 22
    VARCHAR = 23
    BLOB = 24
    BIT = 25
    INTERVAL = 26

    def to_duckdb_type(self) -> duckdb.typing.DuckDBPyType:
        if self == UDFType.SQLNULL:
            return duckdb.typing.SQLNULL
        elif self == UDFType.BOOLEAN:
            return duckdb.typing.BOOLEAN
        elif self == UDFType.TINYINT:
            return duckdb.typing.TINYINT
        elif self == UDFType.UTINYINT:
            return duckdb.typing.UTINYINT
        elif self == UDFType.SMALLINT:
            return duckdb.typing.SMALLINT
        elif self == UDFType.USMALLINT:
            return duckdb.typing.USMALLINT
        elif self == UDFType.INTEGER:
            return duckdb.typing.INTEGER
        elif self == UDFType.UINTEGER:
            return duckdb.typing.UINTEGER
        elif self == UDFType.BIGINT:
            return duckdb.typing.BIGINT
        elif self == UDFType.UBIGINT:
            return duckdb.typing.UBIGINT
        elif self == UDFType.HUGEINT:
            return duckdb.typing.HUGEINT
        elif self == UDFType.UUID:
            return duckdb.typing.UUID
        elif self == UDFType.FLOAT:
            return duckdb.typing.FLOAT
        elif self == UDFType.DOUBLE:
            return duckdb.typing.DOUBLE
        elif self == UDFType.DATE:
            return duckdb.typing.DATE
        elif self == UDFType.TIMESTAMP:
            return duckdb.typing.TIMESTAMP
        elif self == UDFType.TIMESTAMP_MS:
            return duckdb.typing.TIMESTAMP_MS
        elif self == UDFType.TIMESTAMP_NS:
            return duckdb.typing.TIMESTAMP_NS
        elif self == UDFType.TIMESTAMP_S:
            return duckdb.typing.TIMESTAMP_S
        elif self == UDFType.TIME:
            return duckdb.typing.TIME
        elif self == UDFType.TIME_TZ:
            return duckdb.typing.TIME_TZ
        elif self == UDFType.TIMESTAMP_TZ:
            return duckdb.typing.TIMESTAMP_TZ
        elif self == UDFType.VARCHAR:
            return duckdb.typing.VARCHAR
        elif self == UDFType.BLOB:
            return duckdb.typing.BLOB
        elif self == UDFType.BIT:
            return duckdb.typing.BIT
        elif self == UDFType.INTERVAL:
            return duckdb.typing.INTERVAL
        return None


class UDFStructType:
    """
    A wrapper of duckdb.struct_type, eg: UDFStructType({'host': 'VARCHAR', 'path:' 'VARCHAR', 'query': 'VARCHAR'})

    See https://duckdb.org/docs/api/python/types.html#a-field_one-b-field_two--n-field_n
    """

    def __init__(self, fields: Union[Dict[str, str], List[str]]) -> None:
        self.fields = fields

    def to_duckdb_type(self) -> duckdb.typing.DuckDBPyType:
        return duckdb.struct_type(self.fields)


class UDFListType:
    """
    A wrapper of duckdb.list_type, eg: UDFListType(UDFType.INTEGER)

    See https://duckdb.org/docs/api/python/types.html#listchild_type
    """

    def __init__(self, child) -> None:
        self.child = child

    def to_duckdb_type(self) -> duckdb.typing.DuckDBPyType:
        return duckdb.list_type(self.child.to_duckdb_type())


class UDFMapType:
    """
    A wrapper of duckdb.map_type, eg: UDFMapType(UDFType.VARCHAR, UDFType.INTEGER)

    See https://duckdb.org/docs/api/python/types.html#dictkey_type-value_type
    """

    def __init__(self, key, value) -> None:
        self.key = key
        self.value = value

    def to_duckdb_type(self) -> duckdb.typing.DuckDBPyType:
        return duckdb.map_type(self.key.to_duckdb_type(), self.value.to_duckdb_type())


class UDFAnyParameters:
    """
    Accept parameters of any types in UDF.
    """

    def __init__(self) -> None:
        pass

    def to_duckdb_type(self) -> duckdb.typing.DuckDBPyType:
        return None


class UDFContext(object):
    def bind(self, conn: duckdb.DuckDBPyConnection):
        raise NotImplementedError


class PythonUDFContext(UDFContext):
    def __init__(
        self,
        name: str,
        func: Callable,
        params: Optional[List[UDFType]],
        return_type: Optional[UDFType],
        use_arrow_type=False,
    ):
        self.name = name
        self.func = func
        self.params = params
        self.return_type = return_type
        self.use_arrow_type = use_arrow_type

    def __str__(self) -> str:
        return f"{self.name}@{self.func}"

    __repr__ = __str__

    def bind(self, conn: duckdb.DuckDBPyConnection):
        if isinstance(self.params, UDFAnyParameters):
            duckdb_args = self.params.to_duckdb_type()
        else:
            duckdb_args = [arg.to_duckdb_type() for arg in self.params]
        conn.create_function(
            self.name,
            self.func,
            duckdb_args,
            self.return_type.to_duckdb_type(),
            type=("arrow" if self.use_arrow_type else "native"),
        )
        # logger.debug(f"created python udf: {self.name}({self.params}) -> {self.return_type}")


class ExternalModuleContext(UDFContext):
    def __init__(self, name: str, module_path: str) -> None:
        self.name = name
        self.module_path = module_path

    def __str__(self) -> str:
        return f"{self.name}@{self.module_path}"

    __repr__ = __str__

    def bind(self, conn: duckdb.DuckDBPyConnection):
        module_name, _ = os.path.splitext(os.path.basename(self.module_path))
        spec = importlib.util.spec_from_file_location(module_name, self.module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.create_duckdb_udfs(conn)
        # logger.debug(f"loaded external module at {self.module_path}, udf functions: {module.udfs}")


class DuckDbExtensionContext(UDFContext):
    def __init__(self, name: str, extension_path: str) -> None:
        self.name = name
        self.extension_path = extension_path

    def __str__(self) -> str:
        return f"{self.name}@{self.extension_path}"

    __repr__ = __str__

    def bind(self, conn: duckdb.DuckDBPyConnection):
        conn.load_extension(self.extension_path)
        # logger.debug(f"loaded duckdb extension at {self.extension_path}")


@dataclass
class UserDefinedFunction:
    """
    A python user-defined function.
    """

    name: str
    func: Callable
    params: List[UDFType]
    return_type: UDFType
    use_arrow_type: bool

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def udf(
    params: List[UDFType],
    return_type: UDFType,
    use_arrow_type: bool = False,
    name: Optional[str] = None,
) -> Callable[[Callable], UserDefinedFunction]:
    """
    A decorator to define a Python UDF.

    Examples
    --------
    ```
    @udf(params=[UDFType.INTEGER, UDFType.INTEGER], return_type=UDFType.INTEGER)
    def gcd(a: int, b: int) -> int:
      while b:
        a, b = b, a % b
      return a
    ```

    See `Context.create_function` for more details.
    """
    return lambda func: UserDefinedFunction(name or func.__name__, func, params, return_type, use_arrow_type)
