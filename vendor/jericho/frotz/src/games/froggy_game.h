/*
Copyright (C) 2018 Microsoft Corporation

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
*/

#ifndef FROGGY_GAME_H
#define FROGGY_GAME_H

#include <string.h>
#include "frotz.h"
#include "games.h"
#include "frotz_interface.h"

static int froggy_read_word(int address) {
  return (((short) zmp[address]) << 8) | zmp[address + 1];
}

static char* froggy_macro_clean_observation(char* obs) {
  char* prompt = strrchr(obs, '>');
  if (prompt != NULL && prompt > obs) {
    *(prompt - 1) = '\0';
  }
  return obs + 1;
}

#define DEFINE_FROGGY_GAME(                                                \
    prefix, victory_text, loss_text, player_obj, moves_addr, moves_base,   \
    score_addr, score_base, maximum_score, world_objs, player_name,        \
    location_obj, location_name)                                           \
  zword* prefix##_ram_addrs(int *n) {                                      \
    *n = 0;                                                                 \
    return NULL;                                                            \
  }                                                                         \
                                                                            \
  char** prefix##_intro_actions(int *n) {                                   \
    *n = 0;                                                                 \
    return NULL;                                                            \
  }                                                                         \
                                                                            \
  char* prefix##_clean_observation(char* obs) {                             \
    return froggy_macro_clean_observation(obs);                             \
  }                                                                         \
                                                                            \
  int prefix##_victory() {                                                  \
    return strstr(world, victory_text) != NULL;                             \
  }                                                                         \
                                                                            \
  int prefix##_game_over() {                                                \
    return strstr(world, loss_text) != NULL;                                \
  }                                                                         \
                                                                            \
  int prefix##_get_self_object_num() {                                      \
    return player_obj;                                                       \
  }                                                                         \
                                                                            \
  int prefix##_get_moves() {                                                \
    int moves = froggy_read_word(moves_addr) - moves_base;                  \
    return moves > 0 ? moves : 0;                                           \
  }                                                                         \
                                                                            \
  short prefix##_get_score() {                                              \
    if (score_addr >= 0) {                                                   \
      return froggy_read_word(score_addr) - score_base;                     \
    }                                                                       \
    return prefix##_victory() ? maximum_score : 0;                          \
  }                                                                         \
                                                                            \
  int prefix##_max_score() {                                                \
    return maximum_score;                                                    \
  }                                                                         \
                                                                            \
  int prefix##_get_num_world_objs() {                                       \
    return world_objs;                                                       \
  }                                                                         \
                                                                            \
  int prefix##_ignore_moved_obj(zword obj_num, zword dest_num) {            \
    return 0;                                                               \
  }                                                                         \
                                                                            \
  int prefix##_ignore_attr_diff(zword obj_num, zword attr_idx) {            \
    return 0;                                                               \
  }                                                                         \
                                                                            \
  int prefix##_ignore_attr_clr(zword obj_num, zword attr_idx) {             \
    return 0;                                                               \
  }                                                                         \
                                                                            \
  void prefix##_clean_world_objs(zobject* objs) {                           \
    strcpy(objs[player_obj].name, player_name);                             \
    if (location_obj > 0) {                                                 \
      strcpy(objs[location_obj].name, location_name);                       \
    }                                                                       \
  }

#endif
