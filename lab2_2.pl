:- set_prolog_flag(occurs_check, error).
:- set_prolog_stack(global, limit(8 000 000)).
:- set_prolog_stack(global, limit(2 000 000)).

kobieta(X) :- osoba(X), \+ mezczyzna(X).
ojciec(X,Y) :- rodzic(X,Y), mezczyzna(X).
matka(X, Y) :- rodzic(X, Y), kobieta(X).
dziecko(X, Y) :- rodzic(Y, X).
corka(X, Y) :- kobieta(X), dziecko(X,Y).
brat_rodzony(X,Y) :- mezczyzna(X), matka(Z,X), matka(Z, Y), ojciec(W,X), ojciec(W, Y).
brat_przyrodni(X,Y) :- mezczyzna(X), rodzic(Z, X), rodzic(Z, Y), \+ brat_rodzony(X, Y).
kuzyn(X, Y) :- rodzic(A, X), rodzic(B, Y), A \= B, rodzic(C, A), rodzic(C, B).
dziadek_od_strony_ojca(X,Y) :- ojciec(X, Z), ojciec(Z, Y).
dziadek_od_strony_matki(X,Y) :- ojciec(X, Z), matka(Z,Y).
dziadek(X,Y) :- rodzic(X, Z), rodzic(Z, Y), mezczyzna(X).
babcia(X,Y) :- matka(X, Z), rodzic(Z, Y). 
wnuczka(X,Y) :- kobieta(X), parent(Y,Z), parent(Z,X).
przodek_do2pokolenia_wstecz(X,Y) :- rodzic(X, Y); (rodzic(X, Z), rodzic(Z,Y)).
przodek_do2pokolenia_wstecz(X,Y) :- rodzic(X, Y); (rodzic(X, Z), przodek_do2pokolenia_wstecz(Z,Y)).

