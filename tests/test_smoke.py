"""Smoke test: the package imports and the scaffold is wired up."""


def test_package_imports() -> None:
    import pkcar

    assert pkcar.__doc__
