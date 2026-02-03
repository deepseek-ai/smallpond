import unittest
from unittest.mock import patch

import duckdb
import duckdb.sqltypes
import pyarrow.compute as pc

from smallpond.logical.udf import (
    PythonUDFContext,
    UDFAnyParameters,
    UDFListType,
    UDFMapType,
    UDFStructType,
    UDFType,
    UserDefinedFunction,
    _UDFTYPE_TO_DUCKDB,
    udf,
)


class TestUDFTypeEnum(unittest.TestCase):
    """Enum values, stability, and the to_duckdb_type() conversion."""

    # ------------------------------------------------------------------
    # value-stability: every member that existed before UHUGEINT was added
    # must keep its original integer value so that any persisted / pickled
    # enum survives a round-trip.
    # ------------------------------------------------------------------
    ORIGINAL_VALUES = {
        "SQLNULL": 1,
        "BOOLEAN": 2,
        "TINYINT": 3,
        "UTINYINT": 4,
        "SMALLINT": 5,
        "USMALLINT": 6,
        "INTEGER": 7,
        "UINTEGER": 8,
        "BIGINT": 9,
        "UBIGINT": 10,
        "HUGEINT": 11,
        "UUID": 12,
        "FLOAT": 13,
        "DOUBLE": 14,
        "DATE": 15,
        "TIMESTAMP": 16,
        "TIMESTAMP_MS": 17,
        "TIMESTAMP_NS": 18,
        "TIMESTAMP_S": 19,
        "TIME": 20,
        "TIME_TZ": 21,
        "TIMESTAMP_TZ": 22,
        "VARCHAR": 23,
        "BLOB": 24,
        "BIT": 25,
        "INTERVAL": 26,
    }

    def test_original_enum_values_unchanged(self):
        """No pre-existing member may have its integer value changed."""
        for name, expected_value in self.ORIGINAL_VALUES.items():
            with self.subTest(name=name):
                self.assertEqual(UDFType[name].value, expected_value)

    def test_uhugeint_does_not_collide(self):
        """UHUGEINT must have a value that no original member used."""
        self.assertNotIn(UDFType.UHUGEINT.value, self.ORIGINAL_VALUES.values())

    def test_value_12_is_uuid(self):
        """Value 12 must resolve to UUID, not UHUGEINT (regression guard)."""
        self.assertIs(UDFType(12), UDFType.UUID)

    def test_every_member_has_a_mapping(self):
        """Every UDFType member must appear in _UDFTYPE_TO_DUCKDB."""
        for member in UDFType:
            with self.subTest(member=member.name):
                self.assertIn(member, _UDFTYPE_TO_DUCKDB)

    def test_every_member_round_trips(self):
        """to_duckdb_type() returns a non-None duckdb type for every member."""
        for member in UDFType:
            with self.subTest(member=member.name):
                result = member.to_duckdb_type()
                self.assertIsNotNone(result)

    def test_uhugeint_resolves_to_correct_duckdb_type(self):
        self.assertEqual(UDFType.UHUGEINT.to_duckdb_type(), duckdb.sqltypes.UHUGEINT)

    # ------------------------------------------------------------------
    # error path: to_duckdb_type() must raise TypeError (not KeyError)
    # when an enum member is missing from the lookup dict.
    # ------------------------------------------------------------------
    def test_missing_mapping_raises_typeerror(self):
        """Temporarily remove one entry; to_duckdb_type() must raise TypeError."""
        with patch.dict(_UDFTYPE_TO_DUCKDB, {}, clear=True):
            with self.assertRaises(TypeError) as ctx:
                UDFType.INTEGER.to_duckdb_type()
            self.assertIn("INTEGER", str(ctx.exception))


class TestUDFComplexTypes(unittest.TestCase):
    """Struct / List / Map / nested combinations and UDFAnyParameters."""

    def test_struct_type(self):
        t = UDFStructType({"host": "VARCHAR", "port": "INTEGER"}).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_list_of_scalar(self):
        t = UDFListType(UDFType.INTEGER).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_list_of_uhugeint(self):
        t = UDFListType(UDFType.UHUGEINT).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_map_varchar_to_integer(self):
        t = UDFMapType(UDFType.VARCHAR, UDFType.INTEGER).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_map_varchar_to_uhugeint(self):
        t = UDFMapType(UDFType.VARCHAR, UDFType.UHUGEINT).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_nested_list_of_struct(self):
        inner = UDFStructType({"x": "DOUBLE", "y": "DOUBLE"})
        t = UDFListType(inner).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_nested_map_of_list(self):
        inner_list = UDFListType(UDFType.VARCHAR)
        t = UDFMapType(UDFType.INTEGER, inner_list).to_duckdb_type()
        self.assertIsNotNone(t)

    def test_any_parameters_returns_none(self):
        self.assertIsNone(UDFAnyParameters().to_duckdb_type())


class TestUDFDecorator(unittest.TestCase):
    """The @udf decorator wires up UserDefinedFunction correctly."""

    def test_decorator_produces_udf_object(self):
        @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER)
        def inc(x):
            return x + 1

        self.assertIsInstance(inc, UserDefinedFunction)

    def test_decorator_preserves_name(self):
        @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER)
        def my_func(x):
            return x

        self.assertEqual(my_func.name, "my_func")

    def test_decorator_explicit_name_override(self):
        @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER, name="custom_name")
        def my_func(x):
            return x

        self.assertEqual(my_func.name, "custom_name")

    def test_decorated_udf_is_callable(self):
        @udf(params=[UDFType.INTEGER, UDFType.INTEGER], return_type=UDFType.INTEGER)
        def add(a, b):
            return a + b

        self.assertEqual(add(3, 4), 7)

    def test_decorator_stores_arrow_flag(self):
        @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER, use_arrow_type=True)
        def f(x):
            return x

        self.assertTrue(f.use_arrow_type)


class TestPythonUDFContextBind(unittest.TestCase):
    """PythonUDFContext.bind() registers functions that DuckDB can actually call."""

    def _conn(self):
        return duckdb.connect()

    # --- native mode, scalar types ---

    def test_bind_native_integer(self):
        conn = self._conn()

        def double(x):
            return x * 2

        PythonUDFContext("double", double, [UDFType.INTEGER], UDFType.INTEGER).bind(conn)
        self.assertEqual(conn.sql("SELECT double(21)").fetchone()[0], 42)

    def test_bind_native_varchar(self):
        conn = self._conn()

        def shout(s):
            return s.upper()

        PythonUDFContext("shout", shout, [UDFType.VARCHAR], UDFType.VARCHAR).bind(conn)
        self.assertEqual(conn.sql("SELECT shout('hello')").fetchone()[0], "HELLO")

    def test_bind_native_uhugeint(self):
        conn = self._conn()

        def triple(x):
            return x * 3

        PythonUDFContext("triple", triple, [UDFType.UHUGEINT], UDFType.UHUGEINT).bind(conn)
        self.assertEqual(conn.sql("SELECT triple(5::UHUGEINT)").fetchone()[0], 15)

    def test_bind_native_boolean(self):
        conn = self._conn()

        def negate(b):
            return not b

        PythonUDFContext("negate", negate, [UDFType.BOOLEAN], UDFType.BOOLEAN).bind(conn)
        self.assertEqual(conn.sql("SELECT negate(true)").fetchone()[0], False)

    def test_bind_native_double(self):
        conn = self._conn()

        def half(x):
            return x / 2.0

        PythonUDFContext("half", half, [UDFType.DOUBLE], UDFType.DOUBLE).bind(conn)
        self.assertAlmostEqual(conn.sql("SELECT half(7.0)").fetchone()[0], 3.5)

    # --- native mode, complex return types ---

    def test_bind_native_list_return(self):
        conn = self._conn()

        def pair(a, b):
            return [a, b]

        PythonUDFContext("pair", pair, [UDFType.INTEGER, UDFType.INTEGER], UDFListType(UDFType.INTEGER)).bind(conn)
        self.assertEqual(conn.sql("SELECT pair(1, 2)").fetchone()[0], [1, 2])

    def test_bind_native_struct_return(self):
        conn = self._conn()

        def split_kv(s):
            k, v = s.split("=", 1)
            return {"key": k, "value": v}

        PythonUDFContext(
            "split_kv", split_kv, [UDFType.VARCHAR], UDFStructType({"key": "VARCHAR", "value": "VARCHAR"})
        ).bind(conn)
        row = conn.sql("SELECT split_kv('name=alice')").fetchone()[0]
        self.assertEqual(row, {"key": "name", "value": "alice"})

    def test_bind_native_map_return(self):
        conn = self._conn()

        def single_map(k, v):
            return {k: v}

        PythonUDFContext("single_map", single_map, [UDFType.VARCHAR, UDFType.INTEGER], UDFMapType(UDFType.VARCHAR, UDFType.INTEGER)).bind(conn)
        self.assertEqual(conn.sql("SELECT single_map('x', 9)").fetchone()[0], {"x": 9})

    def test_bind_native_list_of_uhugeint(self):
        conn = self._conn()

        def wrap(x):
            return [x]

        PythonUDFContext("wrap", wrap, [UDFType.UHUGEINT], UDFListType(UDFType.UHUGEINT)).bind(conn)
        self.assertEqual(conn.sql("SELECT wrap(42::UHUGEINT)").fetchone()[0], [42])

    # --- arrow mode ---

    def test_bind_arrow_mode(self):
        conn = self._conn()

        def arrow_inc(a):
            return pc.add(a, 1)

        PythonUDFContext("arrow_inc", arrow_inc, [UDFType.INTEGER], UDFType.INTEGER, use_arrow_type=True).bind(conn)
        self.assertEqual(conn.sql("SELECT arrow_inc(10)").fetchone()[0], 11)

    def test_bind_arrow_mode_with_list_return(self):
        conn = self._conn()
        import pyarrow as pa

        def arrow_wrap(a):
            # a is a ChunkedArray of VARCHAR; wrap each element into a single-element list
            lists = [[v.as_py()] for v in a]
            return pa.array(lists, type=pa.list_(pa.string()))

        PythonUDFContext("arrow_wrap", arrow_wrap, [UDFType.VARCHAR], UDFListType(UDFType.VARCHAR), use_arrow_type=True).bind(conn)
        self.assertEqual(conn.sql("SELECT arrow_wrap('hi')").fetchone()[0], ["hi"])

    # --- UDFAnyParameters ---

    def test_bind_any_parameters(self):
        conn = self._conn()

        def echo(x):
            return str(x)

        PythonUDFContext("echo", echo, UDFAnyParameters(), UDFType.VARCHAR).bind(conn)
        self.assertEqual(conn.sql("SELECT echo('test')").fetchone()[0], "test")
        # works with a different type on the same function
        self.assertEqual(conn.sql("SELECT echo(42)").fetchone()[0], "42")

    # --- error propagation: missing mapping ---

    def test_bind_raises_typeerror_for_unmapped_type(self):
        """If a UDFType member has no dict entry, bind must surface TypeError, not KeyError."""
        conn = self._conn()

        def dummy(x):
            return x

        with patch.dict(_UDFTYPE_TO_DUCKDB, {}, clear=True):
            ctx = PythonUDFContext("dummy", dummy, [UDFType.INTEGER], UDFType.INTEGER)
            with self.assertRaises(TypeError):
                ctx.bind(conn)


if __name__ == "__main__":
    unittest.main()
