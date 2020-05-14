### Evolutionary Strategy with Box2D

Specifically, the [Cartpole-v0](https://github.com/openai/gym/wiki/CartPole-v0) environment from [OpenAI's Gym](https://github.com/openai/gym), implemented with with Magnum and Eigen and [Adriel-M's Cartpole-v0 solution](https://gist.github.com/Adriel-M/4daabe115982fe1d9159e730ac3f79a5#file-cartpole-els-py-L121).

```bash
$ esbox2d.exe
...
Mouse to interact
Enter to run
R to reset
D to disable on done
```

Enter runs 100 generations of a population of 10 for 200 steps. R uses the current weights to drive the cartpole.

<br>

### Build

You'll need..

- [Magnum-2019.10](https://github.com/mosra/magnum/releases/tag/v2019.10)
- [Eigen-3.3](https://gitlab.com/libeigen/eigen/-/releases/3.3.7)
- [Box2D-1025f9a](https://github.com/erincatto/box2d/commit/1025f9a10949b963d6311995910bdd04f72dae6c)

```bash
git clone https://github.com/alanjfs/esbox2d.git
cd esbox2d
mkdir build
cd build
cmake ..
cmake --build .
```

> Tested with Visual Studio 2019, open the `CMakeLists.txt` file as-is

<br>

### Problem

It currently isn't learning anything.
