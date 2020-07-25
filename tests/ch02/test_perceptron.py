import src.deep_learning.ch02.perceptron as target 

def test_AND_zero():
    assert 0 == target.AND(1, 2)

def test_AND_one():
    assert 1 == target.AND(0, 1)