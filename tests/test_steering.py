import pytest
import numpy as np
from physics.models.steering.steering import SteeringModel


def test_steering_static():
    """Test that zero steering wheel input results in zero wheel angles."""
    model = SteeringModel()
    delta_l, delta_r = model.calculate_steer_angles(0.0)
    assert pytest.approx(delta_l, abs=1e-7) == 0.0
    assert pytest.approx(delta_r, abs=1e-7) == 0.0


def test_steering_left_turn():
    """Test that a positive yoke angle (left turn) results in positive steering angles."""
    model = SteeringModel()
    delta_l, delta_r = model.calculate_steer_angles(30.0)
    
    # Left turn means positive steer angles for both wheels
    assert delta_l > 0.0
    assert delta_r > 0.0
    
    # Due to Ackermann steering geometry, the inner (left) wheel should steer more than the outer (right) wheel.
    assert delta_l > delta_r


def test_steering_right_turn():
    """Test that a negative yoke angle (right turn) results in negative steering angles."""
    model = SteeringModel()
    delta_l, delta_r = model.calculate_steer_angles(-30.0)
    
    # Right turn means negative steer angles for both wheels
    assert delta_l < 0.0
    assert delta_r < 0.0
    
    # Due to Ackermann steering geometry, the inner (right) wheel should steer more than the outer (left) wheel (magnitude-wise).
    assert abs(delta_r) > abs(delta_l)


def test_steering_limits():
    """Test that extremely large yoke inputs raise appropriate overextension errors."""
    model = SteeringModel()
    
    # Very large angle that would exceed travel limits
    with pytest.raises(ValueError) as excinfo:
        model.calculate_steer_angles(300.0)
    assert "Linkage" in str(excinfo.value)
