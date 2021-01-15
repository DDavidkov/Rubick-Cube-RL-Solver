import numpy as np
from cube_model_naive import Cube

def forward_backward_test():
    cube = Cube()
    success = True
    for i in range(0, cube._num_actions, 2):
        cube.set_random_state()
        current_state = cube._state.copy()
        next_state, _, _ = cube.step(i)
        prev_state, _, _ = cube.step(i + 1)
        if not np.all(prev_state == current_state):
            print("forward-backward test failed at action: {}".format(i))
            success = False
    if success:
        print("forward-backward test passed!")
    return success

def inverse_test(num_test, scramlbe):
    cube = Cube ()
    for _ in range(num_test):
        cube.reset()
        actions = np.random.randint(low=0, high=12, size=scramlbe)
        for a in actions:
            cube.step(a)
        inverse = lambda a: a + (-1) ** (a % 2)
        for a in inverse(actions[::-1]):
            cube.step(a)
        if not cube.is_solved():
            print("reverse test failed!")
            return False
    print("reverse test passed!")
    return True

def main():
    forward_backward_test()
    inverse_test(1000, 6)


if __name__ == "__main__":
    main()