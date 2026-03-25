cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferun.exe run.c win.c
cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferunq.exe runq.c win.c
if exist embedded\stories260K\stories260K_q80.h cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferunq_embedded.exe runq_embedded.c win.c
