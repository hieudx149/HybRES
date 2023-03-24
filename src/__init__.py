import os


def get_root_path():
    """Trả về đường dẫn đến thư mục gốc."""
    root_path = os.path.dirname(os.path.realpath(__file__))
    return root_path