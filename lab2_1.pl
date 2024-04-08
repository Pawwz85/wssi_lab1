:- set_prolog_flag(occurs_check, error).
:- set_prolog_stack(global, limit(8 000 000)).
:- set_prolog_stack(global, limit(2 000 000)).

% Dla czytelności
parent(X, Y) :- f(X, Y).

% a, rodzeństwo
share_parent(X, Y, Z) :- parent(Z, X), parent(Z, Y), X \= Y. % X Y share parent Z.
blood_siblings(X, Y) :- share_parent(X, Y, Z), share_parent(X, Y, W), Z \= W.

% b, kuzyni
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
cousins(X, Y) :- grandparent(Z, X), grandparent(Z, Y).

% c dzielenie wnuka
share_grandchild(X, Y) :- grandparent(X, Z), grandparent(Y, Z), X \= Y.

% d przybrany rodzic
step_parent(X, Y) :- share_parent(Y, half_sibling, regular_parent), f(X, half_sibling). % X is step_parent of Y

% e rodzeństwo pół krwii
half_siblings(X, Y) :- share_parent(X, Y, Z), \+ blood_siblings(X, Y).

% f) father_of_nephew_of(X, Y) - X is father of nephew of Y, also X is not sibling with Y
uncle(X, Y) :- parent(Z, Y), blood_siblings(X,Z).

father_of_nephew_of(X, Y) :- parent(X, Z), uncle(Y, Z), \+ sibling(X, Y).

% g) Habsburgowie
habsburg(X, Y) :- uncle(X, Y), sibling(X, Y).