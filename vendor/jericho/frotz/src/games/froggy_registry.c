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

typedef struct {
  const char* victory_text;
  const char* loss_text;
  int player_obj;
  int moves_addr;
  int moves_base;
  int score_addr;
  int score_base;
  int max_score;
  int world_objs;
  const char* player_name;
  int location_obj;
  const char* location_name;
} froggy_game;

static const froggy_game froggy_games[FROGGY_GAME_COUNT] = {
  [FROGGY_INTERNAL] = {
    "*** You have won! ***", "*** You have died ***",
    20, 19772, 0, -1, 0, 1, 333, "you",
    53, "Township of Sebastian"
  },
  [FROGGY_NOROOM] = {
    "*** You have won ***", "*** You have died ***",
    20, 2557, 0, 2555, 0, 1, 37, "you",
    22, "Darkness"
  },
  [FROGGY_PANCAKE_DETECTIVES] = {
    "Ah-ha! Caught blue-mouthed!", "*** You have died ***",
    41, 5959, 0, -1, 0, 1, 58, "you",
    42, "Kitchen"
  },
  [FROGGY_BANANA] = {
    "*** You have won the drinking contest! ***", "*** You have died ***",
    20, 3833, 0, 3831, 0, 10, 57, "you",
    27, "Happy Parrot Bar"
  },
};

static int current_froggy_game = FROGGY_INTERNAL;

static int froggy_registry_read_word(int address) {
  return (((short) zmp[address]) << 8) | zmp[address + 1];
}

void froggy_set_game(int game) {
  current_froggy_game = game;
}

zword* froggy_ram_addrs(int *n) {
  *n = 0;
  return NULL;
}

char** froggy_intro_actions(int *n) {
  *n = 0;
  return NULL;
}

char* froggy_clean_observation(char* obs) {
  char* prompt = strrchr(obs, '>');
  if (prompt != NULL && prompt > obs) {
    *(prompt - 1) = '\0';
  }
  return obs + 1;
}

int froggy_victory() {
  return strstr(
      world, froggy_games[current_froggy_game].victory_text) != NULL;
}

int froggy_game_over() {
  return strstr(
      world, froggy_games[current_froggy_game].loss_text) != NULL;
}

int froggy_get_self_object_num() {
  return froggy_games[current_froggy_game].player_obj;
}

int froggy_get_moves() {
  const froggy_game* game = &froggy_games[current_froggy_game];
  int moves = froggy_registry_read_word(game->moves_addr) - game->moves_base;
  return moves > 0 ? moves : 0;
}

short froggy_get_score() {
  const froggy_game* game = &froggy_games[current_froggy_game];
  if (game->score_addr >= 0) {
    return froggy_registry_read_word(game->score_addr) - game->score_base;
  }
  return froggy_victory() ? game->max_score : 0;
}

int froggy_max_score() {
  return froggy_games[current_froggy_game].max_score;
}

int froggy_get_num_world_objs() {
  return froggy_games[current_froggy_game].world_objs;
}

int froggy_ignore_moved_obj(zword obj_num, zword dest_num) {
  return 0;
}

int froggy_ignore_attr_diff(zword obj_num, zword attr_idx) {
  return 0;
}

int froggy_ignore_attr_clr(zword obj_num, zword attr_idx) {
  return 0;
}

void froggy_clean_world_objs(zobject* objs) {
  const froggy_game* game = &froggy_games[current_froggy_game];
  strcpy(objs[game->player_obj].name, game->player_name);
  if (game->location_obj > 0) {
    strcpy(objs[game->location_obj].name, game->location_name);
  }
}
