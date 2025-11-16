"""Unit tests for NodesProxy."""

import pytest

from comfy.isolation.proxies.nodes_proxy import NodesProxy
import nodes


class TestNodesProxy:
    """Test NodesProxy base class exposure."""

    @pytest.fixture
    def proxy(self):
        """Create a NodesProxy instance for testing."""
        return NodesProxy()

    def test_preview_image_class_accessible(self, proxy):
        """Verify PreviewImage class is accessible via proxy."""
        assert hasattr(proxy, "PreviewImage"), "PreviewImage not found on proxy"
        assert proxy.PreviewImage is not None, "PreviewImage is None"

    def test_preview_image_is_correct_class(self, proxy):
        """Verify PreviewImage is the actual nodes.PreviewImage class."""
        assert proxy.PreviewImage == nodes.PreviewImage, "PreviewImage class mismatch"

    def test_preview_image_is_instantiable(self, proxy):
        """Verify PreviewImage can be instantiated."""
        instance = proxy.PreviewImage()
        assert instance is not None, "PreviewImage instance is None"
        assert isinstance(instance, nodes.PreviewImage), "PreviewImage instance type mismatch"

    def test_save_image_class_accessible(self, proxy):
        """Verify SaveImage class is accessible via proxy."""
        assert hasattr(proxy, "SaveImage"), "SaveImage not found on proxy"
        assert proxy.SaveImage is not None, "SaveImage is None"

    def test_save_image_is_correct_class(self, proxy):
        """Verify SaveImage is the actual nodes.SaveImage class."""
        assert proxy.SaveImage == nodes.SaveImage, "SaveImage class mismatch"

    def test_save_image_is_instantiable(self, proxy):
        """Verify SaveImage can be instantiated."""
        instance = proxy.SaveImage()
        assert instance is not None, "SaveImage instance is None"
        assert isinstance(instance, nodes.SaveImage), "SaveImage instance type mismatch"

    def test_inheritance_works(self, proxy):
        """Verify a class can inherit from proxied base classes."""
        class TestNode(proxy.PreviewImage):
            pass
        
        instance = TestNode()
        assert isinstance(instance, nodes.PreviewImage), "Inheritance from PreviewImage failed"
