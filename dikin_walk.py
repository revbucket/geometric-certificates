#!/usr/bin/env python3

import argparse
from concurrent import futures

import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

from six.moves import range


def hessian(a, b, x):
    """Return log-barrier Hessian matrix at x."""
    d = (b - a.dot(x))
    s = d ** -2.0
    return a.T.dot(np.diag(s)).dot(a)


def local_norm(h, v):
    """Return the local norm of v based on the given Hessian matrix."""
    return v.T.dot(h).dot(v)


def sample_ellipsoid(e, r):
    """Return a point in the (hyper)ellipsoid uniformly sampled.

    The ellipsoid is defined by the positive definite matrix, ``e``, and
    the radius, ``r``.
    """
    # Generate a point on the sphere surface
    p = np.random.normal(size=e.shape[0])
    p /= np.linalg.norm(p)

    # Scale to a point in the sphere volume
    p *= np.random.uniform() ** (1.0/e.shape[0])

    # Transform to a point in the ellipsoid
    return np.sqrt(r) * np.linalg.cholesky(np.linalg.inv(e)).dot(p)


def ellipsoid_axes(e):
    """Return matrix with columns that are the axes of the ellipsoid."""
    w, v = np.linalg.eigh(e)
    return v.dot(np.diag(w**(-1/2.0)))


def dikin_walk(a, b, x0, r=3/40):
    """Generate points with Dikin walk."""
    x = x0
    h_x = hessian(a, b, x)

    while True:
        if not (a.dot(x) <= b).all():
            print(a.dot(x) - b)
            raise Exception('Invalid state: {}'.format(x))

        if np.random.uniform() < 0.5:
            yield x
            continue

        z = x + sample_ellipsoid(h_x, r)
        h_z = hessian(a, b, z)

        if local_norm(h_z, x - z) > 1.0:
            yield x
            continue

        p = np.sqrt(np.linalg.det(h_z) / np.linalg.det(h_x))
        if p >= 1 or np.random.uniform() < p:
            x = z
            h_x = h_z

        yield x


def hit_and_run(a, b, x0):
    """Generate points with Hit-and-run algorithm."""
    x = x0

    while True:
        if not (a.dot(x) <= b).all():
            print(a.dot(x) - b)
            raise Exception('Invalid state: {}'.format(x))

        # Generate a point on the sphere surface
        d = np.random.normal(size=a.shape[1])
        d /= np.linalg.norm(d)

        # Find closest boundary in the direction
        dist = np.divide(b - a.dot(x), a.dot(d))
        closest = dist[dist > 0].min()

        x += d * closest * np.random.uniform()

        yield x


def chebyshev_center(a, b):
    """Return Chebyshev center of the convex polytope."""
    norm_vector = np.reshape(np.linalg.norm(a, axis=1), (a.shape[0], 1))
    c = np.zeros(a.shape[1] + 1)
    c[-1] = -1
    a_lp = np.hstack((a, norm_vector))
    res = linprog(c, A_ub=a_lp, b_ub=b, bounds=(None, None))
    if not res.success:
        raise Exception('Unable to find Chebyshev center')

    return res.x[:-1]


def collect_chain(sampler, count, burn, thin, *args, **kwargs):
    """Use the given sampler to collect points from a chain.

    Args:
        count: Number of points to collect.
        burn: Number of points to skip at beginning of chain.
        thin: Number of points to take from sampler for every point.
    """
    chain = sampler(*args, **kwargs)
    point = next(chain)
    points = np.empty((count, point.shape[0]))

    for i in range(burn - 1):
        next(chain)

    for i in range(count):
        points[i] = next(chain)
        for _ in range(thin - 1):
            next(chain)

    return points


def main():
    """Entry point."""

    parser = argparse.ArgumentParser(description='Dikin walk test')
    parser.add_argument('--sampler', choices=['dikin', 'hit-and-run'],
                        default='dikin', help='Sampling method to use')
    parser.add_argument('--chains', type=int, default=1,
                        help='Number of chains')
    parser.add_argument('--burn', type=int, default=1000,
                        help='Number of samples to burn')
    parser.add_argument('--thin', type=int, default=10,
                        help='Number of samples to take before using one')
    parser.add_argument('--count', type=int, default=10,
                        help='Stop after taking this many samples')
    args = parser.parse_args()

    # This example is based on a system of linear equalities and
    # inequalities. The convex polytope to sample is the nullspace of the
    # given system.

    # Equalities
    # 1) x3 == 0
    eq = np.array([
        [0, 0, 1]
    ])
    eq_rhs = np.array([
        0
    ])

    # Inequalities
    # 1) -3*x1 - 2*x2 <= -6
    # 2) -x2 <= -1
    # 3) x1 - x2 <= 8
    # 4) -3*x1 + x2 <= 4
    # 5) x1 + 3*x2 <= 22
    # 6) x1 <= 10
    leq = np.array([
        [-3, -2, 0],
        [0, -1, 0],
        [1, -1, 0],
        [-3, 1, 0],
        [1, 3, 0],
        [1, 0, 0],
    ])
    leq_rhs = np.array([
        -6, -1, 8, 4, 22, 10
    ])

    # Find nullspace
    u, s, vh = np.linalg.svd(eq)
    rank = np.sum(s >= 1e-10)
    if rank == 0:
        print('No equality constraints given...')
        nullspace = np.identity(vh.shape[0])
    elif rank == vh.shape[0]:
        raise Exception('Only one solution in null space')
    else:
        nullity = vh.shape[0] - rank
        nullspace = vh[-nullity:].T

    # Polytope parameters
    a = leq.dot(nullspace)
    b = leq_rhs

    print('a')
    print(a.shape)
    print('b')
    print(b.shape)
    # Initial point to start the chains from.
    # Use the Chebyshev center.
    x0 = chebyshev_center(a, b)
    print('Chebyshev center: {}'.format(x0.dot(nullspace.T)))
    print(x0.shape)

    print('A= {}'.format(a))
    print('b= {}'.format(b))
    print('x0= {}'.format(x0))

    chain_count = args.chains
    burn = args.burn
    count = args.count
    thin = args.thin

    if args.sampler == 'dikin':
        sampler = dikin_walk
        dikin_radius = 1
        sampler_args = (dikin_radius,)
    elif args.sampler == 'hit-and-run':
        sampler = hit_and_run
        sampler_args = ()
    else:
        parser.error('Invalid sampler: {}'.format(args.sampler))

    import time
    t0 = time.time()
    chain = collect_chain(sampler, count, burn, thin, a, b, x0, *sampler_args)
    t1 = time.time()
    print('time:', t1-t0)


    # Collect chains in parallel
    with futures.ProcessPoolExecutor() as executor:
        fs = [executor.submit(collect_chain, sampler, count, burn, thin,
                              a, b, x0, *sampler_args)
              for c in range(chain_count)]
        chains = [f.result() for f in futures.as_completed(fs)]

    # Plot chains
    for chain_number, chain in enumerate(chains):
        print('Chain {}/{}'.format(chain_number+1, chain_count))

        points = chain.dot(nullspace.T)
        maxes = points.max(axis=0)
        mins = points.min(axis=0)
        margins = 0.1 * (maxes - mins)
        maxes += margins
        mins -= margins

        fig, ax = plt.subplots()
        ax.set_xlim(mins[0], maxes[0])
        ax.set_ylim(mins[1], maxes[1])

        for i in range(leq.shape[0]):
            if leq[i, 1] != 0:
                y1 = (mins[0] * leq[i, 0] - leq_rhs[i]) / -leq[i, 1]
                y2 = (maxes[0] * leq[i, 0] - leq_rhs[i]) / -leq[i, 1]
                ax.plot([mins[0], maxes[0]], [y1, y2], color='black')
            else:
                x = leq_rhs[i] / leq[i, 0]
                ax.plot([x, x], [mins[1], maxes[1]], color='black')

        ax.plot(points[:, 0], points[:, 1], '.')
        plt.show()

        for i in range(points.shape[1]):
            print('Variable x{}'.format(i))
            fig, ax = plt.subplots()
            ax.plot(np.arange(count), points[:, i])
            plt.show()


if __name__ == '__main__':
    main()
