import takahe

uni = takahe.universe.create('eds')
uni.populate()

z = uni.compute_redshift(4)
print(uni.compute_comoving_distance(z))
