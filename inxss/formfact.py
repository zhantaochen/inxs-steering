from collections import namedtuple

def get_ff_params():
    ff = namedtuple(
        'ff', 
        ['A0', 'a0', 'B0', 'b0', 'C0', 'c0', 'D0',
         'A4', 'a4', 'B4', 'b4', 'C4', 'c4', 'D4']
        )(0.0163, 35.8826, 0.3916, 13.2233, 0.6052, 4.3388, -0.0133,
          -0.3803, 10.4033, 0.2838, 3.3780, 0.2108, 1.1036, 0.0050)
    return ff