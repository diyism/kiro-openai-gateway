# -*- coding: utf-8 -*-

"""
Unit tests for the configuration module.
Verifies loading settings from environment variables.
"""

import pytest
import os
from unittest.mock import patch


class TestLogLevelConfig:
    """Tests for LOG_LEVEL configuration."""
    
    def test_default_log_level_is_info(self):
        """
        What it does: Verifies that LOG_LEVEL defaults to INFO.
        Purpose: Ensure that INFO is used when no environment variable is set.
        
        Note: This test verifies the config.py code logic, not the actual
        value from the .env file. We mock os.getenv to simulate
        the absence of the environment variable.
        """
        print("Setup: Mocking os.getenv for LOG_LEVEL...")
        
        # Create a mock that returns None for LOG_LEVEL (simulating missing variable)
        original_getenv = os.getenv
        
        def mock_getenv(key, default=None):
            if key == "LOG_LEVEL":
                print(f"os.getenv('{key}') -> None (mocked)")
                return default  # Return default, simulating missing variable
            return original_getenv(key, default)
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            # Reload config module with mocked getenv
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'INFO', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "INFO"
        
        # Restore module with real values
        import importlib
        import kiro_gateway.config as config_module
        importlib.reload(config_module)
    
    def test_log_level_from_environment(self):
        """
        What it does: Verifies loading LOG_LEVEL from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting LOG_LEVEL=DEBUG...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'DEBUG', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "DEBUG"
    
    def test_log_level_uppercase_conversion(self):
        """
        What it does: Verifies LOG_LEVEL conversion to uppercase.
        Purpose: Ensure that lowercase value is converted to uppercase.
        """
        print("Setup: Setting LOG_LEVEL=warning (lowercase)...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "warning"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'WARNING', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "WARNING"
    
    def test_log_level_trace(self):
        """
        What it does: Verifies setting LOG_LEVEL=TRACE.
        Purpose: Ensure that TRACE level is supported.
        """
        print("Setup: Setting LOG_LEVEL=TRACE...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "TRACE"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "TRACE"
    
    def test_log_level_error(self):
        """
        What it does: Verifies setting LOG_LEVEL=ERROR.
        Purpose: Ensure that ERROR level is supported.
        """
        print("Setup: Setting LOG_LEVEL=ERROR...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "ERROR"
    
    def test_log_level_critical(self):
        """
        What it does: Verifies setting LOG_LEVEL=CRITICAL.
        Purpose: Ensure that CRITICAL level is supported.
        """
        print("Setup: Setting LOG_LEVEL=CRITICAL...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "CRITICAL"


class TestToolDescriptionMaxLengthConfig:
    """Tests for TOOL_DESCRIPTION_MAX_LENGTH configuration."""
    
    def test_default_tool_description_max_length(self):
        """
        What it does: Verifies the default value for TOOL_DESCRIPTION_MAX_LENGTH.
        Purpose: Ensure that 10000 is used by default.
        """
        print("Setup: Removing TOOL_DESCRIPTION_MAX_LENGTH from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "TOOL_DESCRIPTION_MAX_LENGTH" in os.environ:
                del os.environ["TOOL_DESCRIPTION_MAX_LENGTH"]
            
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 10000
    
    def test_tool_description_max_length_from_environment(self):
        """
        What it does: Verifies loading TOOL_DESCRIPTION_MAX_LENGTH from environment.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=5000...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "5000"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 5000
    
    def test_tool_description_max_length_zero_disables(self):
        """
        What it does: Verifies that 0 disables the feature.
        Purpose: Ensure that TOOL_DESCRIPTION_MAX_LENGTH=0 works.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=0...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "0"}):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 0


class TestTimeoutConfigurationWarning:
    """Tests for _warn_timeout_configuration() function."""
    
    def test_no_warning_when_first_token_less_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is NOT shown with correct configuration.
        Purpose: Ensure that no warning when FIRST_TOKEN_TIMEOUT < STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=15, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "15",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should NOT be shown
            assert "WARNING" not in captured.err
            assert "Suboptimal timeout configuration" not in captured.err
    
    def test_warning_when_first_token_equals_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when timeouts are equal.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT == STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=300, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "300",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
    
    def test_warning_when_first_token_greater_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when FIRST_TOKEN > STREAMING.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT > STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=500, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "500",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
            # Verify that timeout values are mentioned in warning
            assert "500" in captured.err
            assert "300" in captured.err
    
    def test_warning_contains_recommendation(self, capsys):
        """
        What it does: Verifies that warning contains a recommendation.
        Purpose: Ensure that user receives useful information.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=400, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "400",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro_gateway.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should contain recommendation
            assert "Recommendation" in captured.err or "LESS than" in captured.err