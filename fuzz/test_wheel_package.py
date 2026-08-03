from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXPECTED_DATA_MEMBERS = {
    "address_parser/data/__init__.py",
    "address_parser/data/address_parser.preprocessed.v104.pkl",
    "address_parser/data/old_districts.json",
    "address_parser/data/old_provinces.json",
    "address_parser/data/old_wards.json",
    "address_parser/data/provinces.json",
    "address_parser/data/ward_mappings.json",
    "address_parser/data/wards.json",
}
PACKAGING_FILES = (
    "DATA_SOURCES.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "setup.py",
)


class WheelPackageTests(unittest.TestCase):
    def _copy_release_source(self, destination: Path) -> None:
        destination.mkdir()
        for filename in PACKAGING_FILES:
            shutil.copy2(PROJECT_ROOT / filename, destination)
        shutil.copytree(PROJECT_ROOT / "address_parser", destination / "address_parser")
        shutil.copytree(PROJECT_ROOT / "typing", destination / "typing")

    def _build_wheel(self, temp_root: Path, native_mode: str) -> Path:
        source_dir = temp_root / f"source-{native_mode}"
        wheel_dir = temp_root / f"wheel-{native_mode}"
        wheel_dir.mkdir()
        self._copy_release_source(source_dir)

        environment = os.environ.copy()
        environment["VN_ADDRESS_PARSER_NATIVE"] = native_mode
        build_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
                str(source_dir),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            build_result.returncode,
            0,
            build_result.stdout + build_result.stderr,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        return wheels[0]

    def _assert_wheel_members(self, wheel_path: Path, *, expect_native: bool) -> None:
        with zipfile.ZipFile(wheel_path) as wheel_archive:
            members = set(wheel_archive.namelist())
            metadata_member = next(
                member for member in members if member.endswith(".dist-info/METADATA")
            )
            metadata = wheel_archive.read(metadata_member).decode()

        packaged_data = {member for member in members if member.startswith("address_parser/data/")}
        native_members = {
            member
            for member in members
            if member.startswith("address_parser/_native_kernels.")
            and member.endswith((".so", ".pyd"))
        }
        self.assertEqual(packaged_data, EXPECTED_DATA_MEMBERS)
        self.assertIn("address_parser/_native_kernels.pyi", members)
        self.assertIn("address_parser/py.typed", members)
        self.assertFalse(any(member.startswith("rank_bm25-stubs/") for member in members))
        self.assertEqual(bool(native_members), expect_native)
        self.assertIn("Version: 0.2.0", metadata)
        self.assertIn("License-Expression: MIT", metadata)
        self.assertTrue(
            any(member.endswith(".dist-info/licenses/DATA_SOURCES.md") for member in members)
        )
        self.assertTrue(
            any(member.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for member in members)
        )
        self.assertIn(
            "Project-URL: Repository, https://github.com/dathuynh1108/address-parser.git",
            metadata,
        )

    def _install_and_smoke(
        self,
        temp_root: Path,
        wheel_path: Path,
        *,
        expect_native: bool,
    ) -> None:
        install_dir = temp_root / f"site-{'native' if expect_native else 'fallback'}"
        install_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(install_dir),
                str(wheel_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            install_result.returncode,
            0,
            install_result.stdout + install_result.stderr,
        )
        smoke_code = "\n".join(
            [
                "from importlib.metadata import version",
                "from pathlib import Path",
                "import address_parser",
                "from address_parser import (",
                "    AddressParser,",
                "    native_acceleration_available,",
                "    require_native_acceleration,",
                ")",
                "site = Path(__import__('sys').argv[1]).resolve()",
                "expected_native = __import__('sys').argv[2] == 'true'",
                "package = Path(address_parser.__file__).resolve()",
                "assert package.is_relative_to(site), (package, site)",
                "assert version('vn-address-parser') == '0.2.0'",
                "assert native_acceleration_available() is expected_native",
                "if expected_native:",
                "    assert require_native_acceleration() is None",
                "else:",
                "    try:",
                "        require_native_acceleration()",
                "    except RuntimeError:",
                "        pass",
                "    else:",
                "        raise AssertionError('fallback wheel unexpectedly reports native support')",
                "result = AddressParser().process(",
                "    'Số 27 Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội'",
                ")",
                "assert result['format'] == 'old', result",
                "assert result['province'] is not None, result",
                "assert result['province']['name'] == 'Hà Nội', result",
            ]
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(install_dir), sysconfig.get_path("purelib"))
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        smoke_result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                smoke_code,
                str(install_dir),
                str(expect_native).lower(),
            ],
            cwd=temp_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            smoke_result.returncode,
            0,
            smoke_result.stdout + smoke_result.stderr,
        )

    def test_fallback_wheel_contains_runtime_assets_and_installs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="address-parser-fallback-wheel-") as temp_dir:
            temp_root = Path(temp_dir)
            wheel_path = self._build_wheel(temp_root, "disabled")

            self.assertTrue(wheel_path.name.endswith("-py3-none-any.whl"))
            self._assert_wheel_members(wheel_path, expect_native=False)
            self._install_and_smoke(temp_root, wheel_path, expect_native=False)

    def test_required_native_wheel_contains_extension_and_installs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="address-parser-native-wheel-") as temp_dir:
            temp_root = Path(temp_dir)
            wheel_path = self._build_wheel(temp_root, "required")

            self.assertFalse(wheel_path.name.endswith("-py3-none-any.whl"))
            self._assert_wheel_members(wheel_path, expect_native=True)
            self._install_and_smoke(temp_root, wheel_path, expect_native=True)


if __name__ == "__main__":
    unittest.main()
