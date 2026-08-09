/*
Copyright (C) 2018 Microsoft Corporation

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
*/

#include <string.h>
#include "frotz.h"
#include "games.h"
#include "frotz_interface.h"

// Kii!Wii!: https://ifdb.org/viewgame?id=vyhpiu9vlxf64nn0

zword* kiiwi_ram_addrs(int *n) {
  *n = 0;
  return NULL;
}

char** kiiwi_intro_actions(int *n) {
  *n = 0;
  return NULL;
}

char* kiiwi_clean_observation(char* obs) {
  char* prompt = strrchr(obs, '>');
  if (prompt != NULL && prompt > obs) {
    *(prompt - 1) = '\0';
  }
  return obs + 1;
}

int kiiwi_victory() {
  return strstr(world, "*** You have made a friend for life. ***") != NULL;
}

int kiiwi_game_over() {
  return 0;
}

int kiiwi_get_self_object_num() {
  return 42;
}

int kiiwi_get_moves() {
  int turns = (((short) zmp[4883]) << 8) | zmp[4884];
  return turns >= 540 ? turns - 540 : 0;
}

short kiiwi_get_score() {
  return kiiwi_victory() ? 1 : 0;
}

int kiiwi_max_score() {
  return 1;
}

int kiiwi_get_num_world_objs() {
  return 45;
}

int kiiwi_ignore_moved_obj(zword obj_num, zword dest_num) {
  return 0;
}

int kiiwi_ignore_attr_diff(zword obj_num, zword attr_idx) {
  return 0;
}

int kiiwi_ignore_attr_clr(zword obj_num, zword attr_idx) {
  return 0;
}

void kiiwi_clean_world_objs(zobject* objs) {
  strcpy(objs[42].name, "you");
  strcpy(objs[43].name, "Park");
  strcpy(objs[45].name, "bird");
}
