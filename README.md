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

The `Enter` key runs 100 generations for 200 steps in a population of 10. The `R` key resets the simulation, using the current weights to drive the cartpole.

<br>

### Build

<img width=500 src=https://user-images.githubusercontent.com/2152766/81909638-ca648200-95c2-11ea-9e72-7c839a69f7ab.gif>

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
