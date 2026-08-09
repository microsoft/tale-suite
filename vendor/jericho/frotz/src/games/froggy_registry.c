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
  const char* md5;
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
  int victory_moves;
} froggy_game;

static const froggy_game froggy_games[] = {
  [FROGGY_INTERNAL] = {
    "72381AA65F38812AB02A9D1CB4A8CC54",
    "*** You have won! ***", "*** You have died ***",
    20, 19772, 0, -1, 0, 1, 333, "you",
    53, "Township of Sebastian"
  },
  [FROGGY_NOROOM] = {
    "315505CE659AAA5217E4C9F22E390D88",
    "*** You have won ***", "*** You have died ***",
    20, 2557, 0, 2555, 0, 1, 37, "you",
    22, "Darkness"
  },
  [FROGGY_PANCAKE_DETECTIVES] = {
    "24B9DB33BA5F6B7EC7C822F192AE96AA",
    "Ah-ha! Caught blue-mouthed!", "*** You have died ***",
    41, 5959, 0, -1, 0, 1, 58, "you",
    42, "Kitchen"
  },
  [FROGGY_BANANA] = {
    "8415999D3C9F7FF50A0A1339283F3B5D",
    "*** You have won the drinking contest! ***", "*** You have died ***",
    20, 3833, 0, 3831, 0, 10, 57, "you",
    27, "Happy Parrot Bar"
  },
  [FROGGY_MRP] = {
    "4780282041BF6C1F5952A5771B432E43",
    "*** You have won ***", "*** You have died ***",
    20, 10460, 0, -1, 0, 1, 183, "you",
    33, "bed"
  },
  [FROGGY_PUTPBAA] = {
    "5F45F6F8625F7B6304FD56CB3A88F77D",
    "*** You win ***", "*** You have died ***",
    20, 2211, 0, -1, 0, 1, 27, "you",
    25, "The Town Square"
  },
  [FROGGY_ANNOY] = {
    "C9AA2222B3FCB8F32F4587BC8C20E98C",
    "*** You have won ***", "*** You have died ***",
    20, 2032, 0, -1, 0, 1, 28, "you",
    24, "West End"
  },
  [FROGGY_EMPTY_ROOM] = {
    "BEA8643693FC28497BD6AD3D8206E9FA",
    "*** You win ***", "*** You have died ***",
    41, 8590, 0, -1, 0, 1, 85, "you",
    44, "White Room"
  },
  [FROGGY_PARANOIA] = {
    "BAE9BA79E584CCD0673E8EE850562035",
    "*** You have won ***", "*** You have died ***",
    41, 6207, 0, -1, 0, 1, 64, "you",
    42, "Playroom"
  },
  [FROGGY_LUDITE] = {
    "5C00C9753A69F0F200A026212A6D1C2E",
    "*** You have won ***", "*** You have died ***",
    20, 2137, 0, -1, 0, 1, 31, "you",
    25, "The Oven"
  },
  [FROGGY_MINIMALIST] = {
    "3B19CA2D5B211F75F88A14C7DBDD2006",
    "*** You have won ***", "*** You have died ***",
    41, 4381, 0, -1, 0, 1, 42, "you",
    42, "Minimalist prompt"
  },
  [FROGGY_PASS_THE_MILK] = {
    "8C1D5939360084CB1DDB6A413FA5F7FE",
    "*** You passed the milk. ***", "*** You have died ***",
    46, 6406, 0, -1, 0, 1, 59, "you",
    45, "chair"
  },
  [FROGGY_FORMS] = {
    "B4852AC4C00E4CE2D74EFD35CAE6B753",
    "*** You have won ***", "*** You have died ***",
    20, 8983, 0, -1, 0, 1, 178, "you",
    30, "In Your Room"
  },
  [FROGGY_I0] = {
    "DED233E32455E9E40022C128916B45AD",
    "*** You have won ***", "*** You have died ***",
    29, 14590, 0, -1, 0, 1, 309, "you",
    43, "In your car"
  },
  [FROGGY_PEACOCK] = {
    "38EB6EAE52D22A20CA89A27CD88AB9B0",
    "*** You have won ***", "*** You have died ***",
    20, 11073, 0, -1, 0, 1, 138, "you",
    86, "Peacock Chamber"
  },
  [FROGGY_PUTPBAD] = {
    "BEE55B4BDD4096119CC56DD68F13EB8F",
    "*** You escaped Lowell Prison dead in a pine box. ***",
    "*** You have died ***",
    41, 4538, 0, -1, 0, 1, 44, "you",
    42, "Lowell Prison yard"
  },
  [FROGGY_SERVICE] = {
    "BC49BFDE10DF06D574CEF218A158CE02",
    "*** You have won ***", "*** You have died ***",
    20, 2011, 0, 2009, 0, 42, 28, "you",
    26, "Chinese Restaurant"
  },
  [FROGGY_SPOT] = {
    "BD6D1300D20E710A600EEDF60604D7A9",
    "*** You have won ***", "*** You have died ***",
    20, 1897, 0, 1895, 0, 1000, 25, "you",
    24, "In a room with the spot"
  },
  [FROGGY_DAY_IN_LIFE] = {
    "081D4490251AC92B3FAB67546A755DCE",
    "*** YOU WON! Way to clean up. Victory is yours! ***",
    "*** You have died ***",
    45, 8225, 0, 8223, 0, 57, 70, "you",
    46, "Parking lot"
  },
#include "froggy_generated.inc"
};

static int current_froggy_game = FROGGY_INTERNAL;

static int froggy_registry_read_word(int address) {
  return (((short) zmp[address]) << 8) | zmp[address + 1];
}

void froggy_set_game(int game) {
  current_froggy_game = game;
}

int froggy_select_game(const char* md5_hash) {
  size_t game;
  for (game = 0; game < sizeof(froggy_games) / sizeof(*froggy_games); game++) {
    if (strcmp(md5_hash, froggy_games[game].md5) == 0) {
      current_froggy_game = game;
      return 1;
    }
  }
  return 0;
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
  const froggy_game* game = &froggy_games[current_froggy_game];
  int moves = froggy_registry_read_word(game->moves_addr) - game->moves_base;
  return strstr(world, game->victory_text) != NULL
      || (game->victory_moves > 0 && moves >= game->victory_moves);
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
  snprintf(
      objs[game->player_obj].name,
      sizeof(objs[game->player_obj].name),
      "%s",
      game->player_name);
  if (game->location_obj > 0 && game->location_obj != game->player_obj) {
    snprintf(
        objs[game->location_obj].name,
        sizeof(objs[game->location_obj].name),
        "%s",
        game->location_name);
  }
}
