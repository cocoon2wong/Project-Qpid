---
layout: page
title: Project-Qpid
cover-img: /subassets/img/head.png
table-of-contents: true
breadcrumbs: true
---
<!--
 * @Author: Conghao Wong
 * @Date: 2025-04-11 10:13:41
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2025-04-30 10:53:42
 * @Github: https://cocoon2wong.github.io
 * Copyright 2025 Conghao Wong, All Rights Reserved.
-->

## Abstract

This is the homepage of our code repository **Project-Qpid**.
This repository was created to facilitate the construction, training and testing of our trajectory prediction models (using the pytorch backend now).
It includes some basic dataset preprocessing and reading as well as training and testing functions to keep our approaches' code repositories compatible with each other and easy to be managed.
Click the following buttons to learn how it works.

> [!WARNING]
> Documents are still being processed, so many pages are incomplete.
> If you find any errors, you can click the “Edit page” button at the bottom of the page to edit it and start a pull request to help us improve it!
> Thank you!

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="./docs">📖 Documents</a>
    <a class="btn btn-colorful btn-lg" href="{{ site.github.repository_url }}">🛠️ Codes</a>
    <br><br>
</div>

## Getting Started

### Code Entrance

The `qpid` package can be used by adding a simple entrance file, like

```python
import sys
import qpid

if __name__ == '__main__':
    qpid.entrance(sys.argv)
```

This file is usually named as `main.py` in our trajectory prediction models' repos.
You can start training, testing, or visualizing models by adding different terminal args in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.

You can run the simple linear prediction model to verify if your dataset files are properly put and processed (please follow the instructions in [this page](https://projectunpredictable.com/Project-Luna/howToUse/), or the README in different models' repos):

```bash
python main.py --model linear --split zara1
```

Its output may like

```null
>>> [Train Manager]: Test Results
    - ADE(Metrics): 0.6033005118370056 (meter).
    - FDE(Metrics): 1.1830337047576904 (meter).
    - Average Inference Time: 5 ms.
    - Fastest Inference Time: 5 ms.
```

You are totally set if you get similar results!
Otherwise, please check the dataset files and where they are placed.


### Args

You can use the following command to view all supported args:

```bash
python main.py --help
```

For example, it may print documents like

```markdown
...

- `--batch_size` (short for `-bs`):
  Batch size when implementation.
    - Type=`int`, argtype=`dynamic`;
    - The default value is `5000`.

- `--compute_loss`:
  Controls whether to compute losses when testing.
    - Type=`int`, argtype=`temporary`;
    - The default value is `0`.

...
```

Here, we can see several properties of this arg, including

- **Data Type**
  
  Type of the arg's accepted value, which could be one of the basic python types like `int` or `float` or `str`.
  Please note that the type `int` could also be used as indicators for `bool` values, like the above arg `--compute_loss`.
  Under these cases, you can pass values `0` or `1` to disable or enable their corresponding functions, like `--compute_loss 1` or `--compute_loss 0`, or simply using the `--compute_loss` alone to enable it, without other values followed.
  For example,

  ```bash
    python main.py ... --batch_size 1000 --compute_loss
  ```

- **Arg Type**
  
  We have defined three arg types to distinguish the loading priority of different args, including `static`, `dynamic`, and `temporary`.

  - Args with argtype `static` can not be changed once after the training is started.
  When testing the model, the code will not parse these args to overwrite all the saved values.
  These args consist mainly of unchangeable attributes of the trajectory prediction model, such as the dimension of the features.
  - Args with argtype `dynamic` can be re-loaded from the terminal to overwrite the saved values.
    The code will try to first parse inputs from the terminal, and then try to load from the saved JSON file (`args.json` in the weights folder).
    These args are often changeable but have different default values for different models, such as the batch size.
  - Args with argtype `temporary` will not be saved into JSON files.
    The code will parse these args from the terminal at each run.

- **Short Name**

  Short name is the alias of args, which usually starts with one `-`.
  These short names and full names are fully replaceable, but please note the number of `-`.
  For example, `--batch_size 1000` equals to the short-name-value `-bs 1000`, while `--bs 1000` is not a valid arg and the code will not parse it.

- **Default Value**