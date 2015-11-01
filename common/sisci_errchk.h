#ifndef SISCI_ERRCHK_H
#define SISCI_ERRCHK_H

#include "sisci_api.h"
#include "sisci_error.h"

static inline void sisci_check_error(sci_error_t error, const char* file, int line, int terminate)
{
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr, "SISCI error code 0x%x at %s, line %d", error, file, line);

		if (terminate)
		{
			SCITerminate();
			exit(EXIT_FAILURE);
		}
	}
}

#define SISCI_ERROR_CHECK

#ifdef SISCI_ERROR_CHECK
	#define sisci_assert(error) { sisci_check_error(error, __FILE__, __LINE__, 1); }
	#define sisci_check(error) { sisci_check_error(error, __FILE__, __LINE__, 0); }
#else
	#define sisci_assert(error) {}
	#define sisci_check(error) {}
#endif

#endif /* SISCI_ERRCHK_H */
