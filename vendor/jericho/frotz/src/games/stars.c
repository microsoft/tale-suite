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

// The Stars Are Right: https://ifdb.org/viewgame?id=dqklf0b4j4m05960

zword* stars_ram_addrs(int *n) {
  *n = 0;
  return NULL;
}

char** stars_intro_actions(int *n) {
  *n = 0;
  return NULL;
}

char* stars_clean_observation(char* obs) {
  char* prompt = strrchr(obs, '>');
  if (prompt != NULL && prompt > obs) {
    *(prompt - 1) = '\0';
  }
  return obs + 1;
}

int stars_victory() {
  return strstr(world, "*** Christmas has won ***") != NULL;
}

int stars_game_over() {
  return strstr(world, "*** You have died ***") != NULL;
}

int stars_get_self_object_num() {
  return 20;
}

int stars_get_moves() {
  return (((short) zmp[2640]) << 8) | zmp[2641];
}

short stars_get_score() {
  return stars_victory() ? 1 : 0;
}

int stars_max_score() {
  return 1;
}

int stars_get_num_world_objs() {
  return 38;
}

int stars_ignore_moved_obj(zword obj_num, zword dest_num) {
  return 0;
}

int stars_ignore_attr_diff(zword obj_num, zword attr_idx) {
  return 0;
}

int stars_ignore_attr_clr(zword obj_num, zword attr_idx) {
  return 0;
}

void stars_clean_world_objs(zobject* objs) {
}
