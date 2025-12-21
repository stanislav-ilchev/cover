/*   tables.h
**
**   This file contains declarations needed when calculating the neighborhood
**   and coverings data tables. The declarations for using those tables are
**   also in this file.
**
*/


#ifndef _tables_
#define _tables_


/* the ranks of the k-sets of the solution */
extern rankType *kset;
/* bitmasks for the k-sets of the solution */
extern maskType *ksetMask;

/* parameters used during the dynamic allocation of the tables */
extern int neighborLen, coverLen, coveredLen;

/* neighbor and coverings data tables */
extern rankType *neighbors, *coverings;
/* cached coverings for on-the-fly mode */
extern rankType *ksetCoverings;
extern int onTheFlyCache;

/* counts of covering k-sets with a given m-set */
extern coveredType *covered;

/* costs and cost differences associated to a m-set covered x times */
extern costType *costs;
extern costDType *costds;

void freeTables(void);
void computeTables(int tl, int kl, int ml, int vl);
void bIs(int bl);
void calculateOneCovering(rankType kRank, rankType *buf);
void calculateOneCoveringMask(maskType mask, rankType *buf);
void calculateOneCoveringMaskFast(maskType mask, rankType *buf);
void sortSolution(void);

#endif
