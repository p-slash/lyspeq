$(DEPDIR)/%.d: $(COREDIR)/%.cpp
	@mkdir -p $(DEPDIR)
	$(CXX) -MM -MG $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(COREDIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

-include $(patsubst %, $(DEPDIR)/%.d, $(basename $(notdir $(CORESOURCES))))
#$(CXX) -MM -MG $< | sed 's/fftw3.h// ; s/gsl.*.h // ; s/gsl.*.h//' > $@.$$$$; \