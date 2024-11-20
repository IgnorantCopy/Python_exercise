unique([]).
unique([Head|Tail]) :- \+ member(Head, Tail), unique(Tail).
check_all([]).
check_all([Head|Tail]) :- unique(Head), check_all(Tail).

is_in_range([], _, _).
is_in_range([Head|Tail], L, R) :- between(L, R, Head), is_in_range(Tail, L, R).
is_all_in_range([]).
is_all_in_range([Head|Tail]) :- is_in_range(Head, 1, 9), is_all_in_range(Tail).

solve(Input, Output) :-
    Input = [[N11, N12, N13, N14, N15, N16, N17, N18, N19],
             [N21, N22, N23, N24, N25, N26, N27, N28, N29],
             [N31, N32, N33, N34, N35, N36, N37, N38, N39],
             [N41, N42, N43, N44, N45, N46, N47, N48, N49],
             [N51, N52, N53, N54, N55, N56, N57, N58, N59],
             [N61, N62, N63, N64, N65, N66, N67, N68, N69],
             [N71, N72, N73, N74, N75, N76, N77, N78, N79],
             [N81, N82, N83, N84, N85, N86, N87, N88, N89],
             [N91, N92, N93, N94, N95, N96, N97, N98, N99]],

    R1 = [ N11, N12, N13, N14, N15, N16, N17, N18, N19 ],
    R2 = [ N21, N22, N23, N24, N25, N26, N27, N28, N29 ],
    R3 = [ N31, N32, N33, N34, N35, N36, N37, N38, N39 ],
    R4 = [ N41, N42, N43, N44, N45, N46, N47, N48, N49 ],
    R5 = [ N51, N52, N53, N54, N55, N56, N57, N58, N59 ],
    R6 = [ N61, N62, N63, N64, N65, N66, N67, N68, N69 ],
    R7 = [ N71, N72, N73, N74, N75, N76, N77, N78, N79 ],
    R8 = [ N81, N82, N83, N84, N85, N86, N87, N88, N89 ],
    R9 = [ N91, N92, N93, N94, N95, N96, N97, N98, N99 ],

    C1 = [ N11, N21, N31, N41, N51, N61, N71, N81, N91 ],
    C2 = [ N12, N22, N32, N42, N52, N62, N72, N82, N92 ],
    C3 = [ N13, N23, N33, N43, N53, N63, N73, N83, N93 ],
    C4 = [ N14, N24, N34, N44, N54, N64, N74, N84, N94 ],
    C5 = [ N15, N25, N35, N45, N55, N65, N75, N85, N95 ],
    C6 = [ N16, N26, N36, N46, N56, N66, N76, N86, N96 ],
    C7 = [ N17, N27, N37, N47, N57, N67, N77, N87, N97 ],
    C8 = [ N18, N28, N38, N48, N58, N68, N78, N88, N98 ],
    C9 = [ N19, N29, N39, N49, N59, N69, N79, N89, N99 ],

    B1 = [ N11, N12, N13, N21, N22, N23, N31, N32, N33 ],
    B2 = [ N14, N15, N16, N24, N25, N26, N34, N35, N36 ],
    B3 = [ N17, N18, N19, N27, N28, N29, N37, N38, N39 ],
    B4 = [ N41, N42, N43, N51, N52, N53, N61, N62, N63 ],
    B5 = [ N44, N45, N46, N54, N55, N56, N64, N65, N66 ],
    B6 = [ N47, N48, N49, N57, N58, N59, N67, N68, N69 ],
    B7 = [ N71, N72, N73, N81, N82, N83, N91, N92, N93 ],
    B8 = [ N74, N75, N76, N84, N85, N86, N94, N95, N96 ],
    B9 = [ N77, N78, N79, N87, N88, N89, N97, N98, N99 ],

    is_all_in_range(Input),
    check_all([R1, R2, R3, R4, R5, R6, R7, R8, R9, C1, C2, C3, C4, C5, C6, C7, C8, C9, B1, B2, B3, B4, B5, B6, B7, B8, B9]),
    Output = Input.
