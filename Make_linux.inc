CC = gcc
CXX = mpic++
CXXFLAGS+=$(mpic++ --showme:compile)
LIBS+=$(mpic++ --showme:link)