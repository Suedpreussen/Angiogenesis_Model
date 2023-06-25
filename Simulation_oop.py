import angiogenesis_oop as an
import time
import cProfile as profile

# In outer section of code
pr = profile.Profile()
pr.enable()


start_time = time.time()
'---------------object oriented approach--------------------'
# instantiate Model
square_model = an.Model(23, 'square', 'square')
#print(square_model.__dict__)

# N should be divisible by 16 or 32
parameters_set = {'a': 3.1, 'b': 4.5, 'gamma': 2/3, 'delta': 2.01, 'nu': 1.1, 'c': 0.001, 'r': 2.2, 'dt': 0.01, 'N': 1}
square_model.run_simulation("graphs", **parameters_set, is_scaled=True)

print(f"time elapsed for OOP: {(time.time() - start_time):.4f}s")


pr.disable()
pr.dump_stats('profile3.pstat')
