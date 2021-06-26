# CRYPTO

## Research
- The parties first secret-share their inputs; i.e. input xi is shared so that ∑jxij=xi and party Pj holds xij (and Pi which provides input is included in this sharing, even though it knows the sum).
- The parties perform additions and multiplications on these secret values by local computations and communication of certain values (in methods specified below). By construction, the result of performing an operation is automatically shared amongst the parties (i.e. with no further communication or computation).
- Finally, the parties 'open' the result of the circuit evaluation. This last step involves each party sending their 'final' share to every other party (and also performing a check that no errors were introduced by the adversary along the way).

