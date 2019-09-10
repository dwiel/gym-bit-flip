import numpy as np
import copy
import pytest

from gym_bit_flip import BitFlip


def test_bitflip_1():
    bit_flip = BitFlip(1)
    bit_flip.state = np.array([0])
    bit_flip.step(0)

    np.testing.assert_array_equal(bit_flip.state, np.array([1]))


def test_bitflip_2():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip.step(1)

    np.testing.assert_array_equal(bit_flip.state, np.array([0, 1]))


def test_bitflip_not_terminate_long():
    bit_flip = BitFlip(256)
    assert bit_flip._terminate() is False


def test_bitflip_not_terminate_short():
    bit_flip = BitFlip(2)
    for _ in range(16):
        assert bit_flip._terminate() is False
        bit_flip.reset()


def test_bitflip_bit_length_0():
    with pytest.raises(ValueError):
        BitFlip(bit_length=0)


def test_reward():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip.goal = np.array([1, 0])
    _, reward, _, _ = bit_flip.step(1)

    assert reward == -1


def test_reward_expected_success():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip.goal = np.array([1, 0])
    _, reward, terminate, _ = bit_flip.step(0)

    assert reward == 0
    assert terminate == True


def test_reward_expected_success_long():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip.goal = np.array([1, 1])

    _, reward, terminate, _ = bit_flip.step(0)
    assert reward == -1
    assert terminate == False

    _, reward, terminate, _ = bit_flip.step(1)
    assert reward == 0
    assert terminate == True


def test_reward_expected_failure():
    bit_flip = BitFlip(256)
    _, reward, _, _ = bit_flip.step(0)

    assert reward == -1


def test_mean_zero():
    bit_flip = BitFlip(mean_zero=True)
    state, _, _, _ = bit_flip.step(0)

    assert 1 in state["state"]
    assert -1 in state["state"]
    assert 1 in state["goal"]
    assert -1 in state["goal"]


def test_observation_copy():
    """
    in many use cases, the previous observation is kept around after a step has
    taken place so that the observation before and after the step can be
    considered together. This test makes sure that taking a step doesn't modify
    via side effect previously returned observations
    """
    bit_flip = BitFlip(2)

    observation, _, _, _ = bit_flip.step(0)
    observation_copy = copy.deepcopy(observation)

    print(observation)
    print(observation_copy)

    # buggy code might result in side effects changing observation (but not the
    # copy) here
    bit_flip.step(0)

    print(observation)
    print(observation_copy)

    np.testing.assert_array_equal(observation['state'], observation_copy['state'])
