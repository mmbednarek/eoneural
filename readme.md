# eoNeural (eon)

Simple cli application to do_train_iteration neural networks.

## Compilation

Standard compliation procedure

```bash
make
sudo make install
```

## Training an XOR function

```bash
eon create 2 2 1
eon pack - data
0 0 0
0 1 1
1 0 1
1 1 0
^D
eon do_train_iteration out.net data
```

## Basic commands

### Creating a neural network

```bash
eon create [INPUT LEYER NEURONS] [HIDDEN LEYERS NEURONS] [OUTPUT LAYER NEURONS] -o [OUTPUT NETWORK]
```

In no output file specified the program will output to *out.net*.

**Example**

```bash
eon create 12 6 9 -o foo.net
```

You can specify the activation function with **-a** option. The default is sigmoid
Currently avaliable activation functions are:

+ identity
+ sigmoid
+ tanh
+ atan
+ gaussian

### Displaying the network info

```bash
eon show out.net
```

### Packing data

Assuming that we have a file *data.txt*
with content

```
1 2 3
4 5 6
```

We can pack the numbers into a data file with this command.

```bash
eon pack data.txt data.bin
```

It is sometimes useful to pack the data from the standard stream

```bash
eon pack - data.bin
```

### Passing values though network

To pass basic number set

```bash
eon pass out.net 0 0 1
```

We can also test the result of training data

```bash
eon pass out.net -t datafile
```

We simply pass data from file

```bash
eon pass out.net -d datafile
```

### Traing the network

Basic training command is

```bash
eon do_train_iteration out.net data
```

If we don't want to lose the source network file
we can specify the output as a third argument or with -o option.

```bash
eon do_train_iteration out.net data trained.net
```

The network is being trained until the RMSE is less or equal to 0.01
Other goal can be specified with the **-r** option.
Alternatively you can pass the target epoch number with **-e** option.
Learning coefficient can be specified with the **-l** option. (The default is 0.7)
Momentum can be specified with the **-m** option. (The default is 0.3)

**Example**

```bash
eon do_train_iteration foo.net data trained.net -r 0.0005 -l 0.6 -m 0.4
```

