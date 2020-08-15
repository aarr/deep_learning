import src.deep_learning.ch02.perceptron as target 

def test_AND():
    assert 0 == target.AND(0, 0)
    assert 0 == target.AND(0, 1)
    assert 0 == target.AND(1, 0)
    assert 1 == target.AND(1, 1)

def test_AND_BIAS():
    assert 0 == target.AND_BIAS(0, 0)
    assert 0 == target.AND_BIAS(0, 1)
    assert 0 == target.AND_BIAS(1, 0)
    assert 1 == target.AND_BIAS(1, 1)

def test_XAND():
    assert 1 == target.NAND(0, 0)
    assert 1 == target.NAND(0, 1)
    assert 1 == target.NAND(1, 0)
    assert 0 == target.NAND(1, 1)

def test_OR():
    assert 0 == target.OR(0, 0)
    assert 1 == target.OR(0, 1)
    assert 1 == target.OR(1, 0)
    assert 1 == target.OR(1, 1)

def test_XOR():
    assert 0 == target.XOR(0, 0)
    assert 1 == target.XOR(0, 1)
    assert 1 == target.XOR(1, 0)
    assert 0 == target.XOR(1, 1)