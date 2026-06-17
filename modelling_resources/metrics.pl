#!/usr/bin/env swipl

:- use_module(library(pio)).
:- use_module(library(dcg/basics)).

:- initialization(main,main).

main(Argv) :-
    Argv = [Reffile,Testfile],
    !,
    write_scores(Reffile,Testfile).
main(_) :-
    writeln("Specify arguments as \n ./metrics.pl -- arg1 arg2 \n You must specify precisely two arguments, the first one being the ground truth reference and the second one the file whose metrics are to be computed.").

pairs([])           --> eos.
pairs([Rule-Prob|Lines]) --> number(Prob), whites, ":",":", whites, string(Rule),"\n", pairs(Lines).

tp(Refs,Test,TP) :-
    tp(Refs,Test,0,TP).

tp([],[],TP,TP) :-
    !.
tp([Rule-RefProb|Refs],[Rule-TestProb|Tests],Aggr,TP) :-
    NewAggr is Aggr + min(RefProb,TestProb),
    tp(Refs,Tests,NewAggr,TP).

fp(Refs,Test,FP) :-
    fp(Refs,Test,0,FP).

fp([],[],FP,FP) :-
    !.
fp([Rule-RefProb|Refs],[Rule-TestProb|Tests],Aggr,FP) :-
    NewAggr is Aggr + max(0, TestProb - RefProb),
    fp(Refs,Tests,NewAggr,FP).

positives(Refs,Pos) :-
    positives(Refs,0,Pos).
positives([],Pos,Pos).
positives([_-RefProb|Refs],Acc,Pos) :-
    NewAcc is Acc + RefProb,
    positives(Refs,NewAcc,Pos).

negatives(Refs,Neg) :-
    positives(Refs,Pos),
    length(Refs,N),
    Neg is N - Pos.


tn(Refs,Test,TN) :-
    fp(Refs,Test,FP),
    negatives(Refs,N),
    TN is N - FP.

fn(Refs,Test,FN) :-
    tp(Refs,Test,TP),
    positives(Refs,N),
    FN is N - TP.

accuracy(Refs,Test,Acc) :-
    tp(Refs,Test,TP),
    tn(Refs,Test,TN),
    length(Refs,N),
    Acc is (TP + TN)/N.

precision(Refs,Test,Prec) :-
    tp(Refs,Test,TP),
    fp(Refs,Test,FP),
    Prec is TP/(TP + FP).

recall(Refs,Test,Recall) :-
    tp(Refs,Test,TP),
    fn(Refs,Test,FN),
    Recall is TP/(TP + FN).

write_scores(Refs,Test,Acc,Prec,Recall) :-
    accuracy(Refs,Test,Acc),
    precision(Refs,Test,Prec),
    recall(Refs,Test,Recall),
    write("Accuracy: "),
    writeln(Acc),
    write("Precision: "),
    writeln(Prec),
    write("Recall: "),
    writeln(Recall).

write_scores(Reffile,Testfile) :-
    phrase_from_file(pairs(Refs),Reffile),
    phrase_from_file(pairs(Test),Testfile),
    write_scores(Refs,Test,_,_,_).


