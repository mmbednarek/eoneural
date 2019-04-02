CC=clang
CFLAGS=-lm -Wall -g
SRC = main.c neuron.c interface.c network.c activation.c
OUT=eon
PREFIX=/usr
MANPREFIX=/usr/share/man

all: $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(OUT)

install: all
	cp -f $(OUT) $(PREFIX)/bin
	chmod 755 $(PREFIX)/bin/nn
	cp -f nn.1.gz $(MANPREFIX)/man1
	chmod 644 $(MANPREFIX)/man1/nn.1.gz
