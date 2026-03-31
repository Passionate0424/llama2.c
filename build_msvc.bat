cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferun.exe run.c win.c
cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferunq.exe runq.c win.c
if exist embedded\stories260K\stories260K_q80.h cl.exe /nologo /fp:fast /Ox /openmp /I. /Ferunq_embedded.exe runq_embedded.c win.c
cl.exe /nologo /fp:fast /Ox /I. /Ideploy\include /DRUNQ_DEPLOY_CPU /Ferunq_deploy_cpu.exe deploy\src\runq_deploy.c deploy\src\runtime_frontend.c deploy\src\runtime_assets.c deploy\src\runtime_common.c deploy\src\runtime_backend_swref.c deploy\src\runtime_backend_hwstub.c deploy\src\runtime_hw_adapter.c
cl.exe /nologo /fp:fast /Ox /I. /Ideploy\include /DRUNQ_DEPLOY_HW /Ferunq_deploy_hw.exe deploy\src\runq_deploy.c deploy\src\runtime_frontend.c deploy\src\runtime_assets.c deploy\src\runtime_common.c deploy\src\runtime_backend_swref.c deploy\src\runtime_backend_hwstub.c deploy\src\runtime_hw_adapter.c
cl.exe /nologo /fp:fast /Ox /I. /Ideploy\include /Ferunq_verify.exe deploy\src\runq_verify.c deploy\src\runtime_verify.c deploy\src\runtime_frontend.c deploy\src\runtime_assets.c deploy\src\runtime_common.c deploy\src\runtime_backend_swref.c deploy\src\runtime_backend_hwstub.c deploy\src\runtime_hw_adapter.c
