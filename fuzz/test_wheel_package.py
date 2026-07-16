import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXPECTED_DATA_MEMBERS = {
    "address_parser/data/__init__.py",
    "address_parser/data/address_parser.preprocessed.v104.pkl",
    "address_parser/data/administrative_units.json",
    "address_parser/data/old_administrative_units.json",
    "address_parser/data/old_districts.json",
    "address_parser/data/old_provinces.json",
    "address_parser/data/old_wards.json",
    "address_parser/data/provinces.json",
    "address_parser/data/ward_mappings.json",
    "address_parser/data/ward_mappings_after_enrich.json",
    "address_parser/data/ward_mappings_enrich.json",
    "address_parser/data/ward_mappings_source.json",
    "address_parser/data/wards.json",
}


class WheelPackageTests(unittest.TestCase):
    def test_wheel_contains_runtime_assets_and_imports_outside_source_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="address-parser-wheel-test-") as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            wheel_dir = temp_root / "wheel"
            install_dir = temp_root / "site"
            source_dir.mkdir()
            wheel_dir.mkdir()

            shutil.copy2(PROJECT_ROOT / "pyproject.toml", source_dir)
            shutil.copy2(PROJECT_ROOT / "README.md", source_dir)
            shutil.copytree(PROJECT_ROOT / "address_parser", source_dir / "address_parser")

            subprocess.run(
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
                check=True,
                capture_output=True,
                text=True,
            )
            wheels = list(wheel_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            wheel_path = wheels[0]

            with zipfile.ZipFile(wheel_path) as wheel_archive:
                members = set(wheel_archive.namelist())
            packaged_data = {
                member for member in members if member.startswith("address_parser/data/")
            }
            self.assertEqual(packaged_data, EXPECTED_DATA_MEMBERS)
            self.assertIn("address_parser/py.typed", members)
            self.assertFalse(any(member.startswith("rank_bm25-stubs/") for member in members))

            subprocess.run(
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
                check=True,
                capture_output=True,
                text=True,
            )
            smoke_code = "\n".join(
                [
                    "from pathlib import Path",
                    "import address_parser",
                    "from address_parser import AddressParser",
                    "site = Path(__import__('sys').argv[1]).resolve()",
                    "package = Path(address_parser.__file__).resolve()",
                    "assert package.is_relative_to(site), (package, site)",
                    "result = AddressParser().process(",
                    "    'Số 27 Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội'",
                    ")",
                    "assert result['format'] == 'old', result",
                    "assert result['province'] is not None, result",
                    "assert result['province']['name'] == 'Hà Nội', result",
                ]
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(install_dir)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                [sys.executable, "-c", smoke_code, str(install_dir)],
                cwd=temp_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
