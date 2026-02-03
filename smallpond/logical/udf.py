import importlib
import os.path
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import duckdb
import duckdb.sqltypes


class UDFType(Enum):
    """
    A wrapper of duckdb.sqltypes.DuckDBPyType

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
    UHUGEINT = 27

    def to_duckdb_type(self) -> duckdb.sqltypes.DuckDBPyType:
        duckdb_type = _UDFTYPE_TO_DUCKDB.get(self)
        if duckdb_type is None:
            raise TypeError(f"UDFType {self.name} has no corresponding duckdb type mapping")
        return duckdb_type


# Mapping from UDFType enum members to duckdb.sqltypes constants.
# Adding a new type here is sufficient; no other code needs to change.
_UDFTYPE_TO_DUCKDB = {
    UDFType.SQLNULL: duckdb.sqltypes.SQLNULL,
    UDFType.BOOLEAN: duckdb.sqltypes.BOOLEAN,
    UDFType.TINYINT: duckdb.sqltypes.TINYINT,
    UDFType.UTINYINT: duckdb.sqltypes.UTINYINT,
    UDFType.SMALLINT: duckdb.sqltypes.SMALLINT,
    UDFType.USMALLINT: duckdb.sqltypes.USMALLINT,
    UDFType.INTEGER: duckdb.sqltypes.INTEGER,
    UDFType.UINTEGER: duckdb.sqltypes.UINTEGER,
    UDFType.BIGINT: duckdb.sqltypes.BIGINT,
    UDFType.UBIGINT: duckdb.sqltypes.UBIGINT,
    UDFType.HUGEINT: duckdb.sqltypes.HUGEINT,
    UDFType.UHUGEINT: duckdb.sqltypes.UHUGEINT,
    UDFType.UUID: duckdb.sqltypes.UUID,
    UDFType.FLOAT: duckdb.sqltypes.FLOAT,
    UDFType.DOUBLE: duckdb.sqltypes.DOUBLE,
    UDFType.DATE: duckdb.sqltypes.DATE,
    UDFType.TIMESTAMP: duckdb.sqltypes.TIMESTAMP,
    UDFType.TIMESTAMP_MS: duckdb.sqltypes.TIMESTAMP_MS,
    UDFType.TIMESTAMP_NS: duckdb.sqltypes.TIMESTAMP_NS,
    UDFType.TIMESTAMP_S: duckdb.sqltypes.TIMESTAMP_S,
    UDFType.TIME: duckdb.sqltypes.TIME,
    UDFType.TIME_TZ: duckdb.sqltypes.TIME_TZ,
    UDFType.TIMESTAMP_TZ: duckdb.sqltypes.TIMESTAMP_TZ,
    UDFType.VARCHAR: duckdb.sqltypes.VARCHAR,
    UDFType.BLOB: duckdb.sqltypes.BLOB,
    UDFType.BIT: duckdb.sqltypes.BIT,
    UDFType.INTERVAL: duckdb.sqltypes.INTERVAL,
}


class UDFStructType:
    """
    A wrapper of duckdb.struct_type, eg: UDFStructType({'host': 'VARCHAR', 'path:' 'VARCHAR', 'query': 'VARCHAR'})

    See https://duckdb.org/docs/api/python/types.html#a-field_one-b-field_two--n-field_n
    """

    def __init__(self, fields: Union[Dict[str, str], List[str]]) -> None:
        self.fields = fields

    def to_duckdb_type(self) -> duckdb.sqltypes.DuckDBPyType:
        return duckdb.struct_type(self.fields)


class UDFListType:
    """
    A wrapper of duckdb.list_type, eg: UDFListType(UDFType.INTEGER)

    See https://duckdb.org/docs/api/python/types.html#listchild_type
    """

    def __init__(self, child) -> None:
        self.child = child

    def to_duckdb_type(self) -> duckdb.sqltypes.DuckDBPyType:
        return duckdb.list_type(self.child.to_duckdb_type())


class UDFMapType:
    """
    A wrapper of duckdb.map_type, eg: UDFMapType(UDFType.VARCHAR, UDFType.INTEGER)

    See https://duckdb.org/docs/api/python/types.html#dictkey_type-value_type
    """

    def __init__(self, key, value) -> None:
        self.key = key
        self.value = value

    def to_duckdb_type(self) -> duckdb.sqltypes.DuckDBPyType:
        return duckdb.map_type(self.key.to_duckdb_type(), self.value.to_duckdb_type())


class UDFAnyParameters:
    """
    Accept parameters of any types in UDF.
    """

    def __init__(self) -> None:
        pass

    def to_duckdb_type(self) -> duckdb.sqltypes.DuckDBPyType:
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
